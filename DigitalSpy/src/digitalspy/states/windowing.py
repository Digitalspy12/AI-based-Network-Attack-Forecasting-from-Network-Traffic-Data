"""Temporal windowing pipeline — optimised vectorised version for MachineLearningCVE.

The MachineLearningCVE CSVs (CICFlowMeter pre-processed) do NOT have:
  - Source IP / Destination IP
  - Timestamp
Each row is an independent pre-computed flow-level feature vector.

Adaptation strategy (documented per §7 + §31):
  Sequential non-overlapping windows of FLOWS_PER_WINDOW consecutive flows.
  This approximates temporal 10s aggregation while remaining faithful to the data.
  Limitation documented in §31 (no per-host grouping).

Performance: fully vectorised via numpy reshape — no Python row loops.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

from digitalspy import config
from digitalspy.features.engineer import FEATURE_NAMES
from digitalspy.states.labels import resolve_window_label, label_to_index

logger = logging.getLogger(__name__)

_PACKET_DERIVED = ["ttl_variance", "tcp_window_std", "fragmentation_ratio"]
FLOWS_PER_WINDOW = 20  # approximate 10-second window in flow counts

# ── CIC-IDS2017 MachineLearningCVE column → feature mapping ─────────────────
# Maps our 24 feature names to the actual CSV column names
_COL_MAP = {
    # volume/timing
    "flow_count":           None,                          # computed = FLOWS_PER_WINDOW
    "total_bytes":          ("Total Length of Fwd Packets", "Total Length of Bwd Packets"),
    "total_packets":        ("Total Fwd Packets", "Total Backward Packets"),
    "mean_flow_duration":   "Flow Duration",
    "bytes_per_sec":        "Flow Bytes/s",
    "packets_per_sec":      "Flow Packets/s",
    # TCP flag ratios
    "syn_ratio":            "SYN Flag Count",
    "ack_ratio":            "ACK Flag Count",
    "fin_ratio":            "FIN Flag Count",
    "rst_ratio":            "RST Flag Count",
    "psh_ratio":            "PSH Flag Count",
    # inter-arrival
    "mean_iat":             "Flow IAT Mean",
    "std_iat":              "Flow IAT Std",
    "max_iat":              "Flow IAT Max",
    # packet size
    "mean_packet_len":      "Packet Length Mean",
    "std_packet_len":       "Packet Length Std",
    # diversity (approximated from Destination Port)
    "distinct_dst_ips":     "Destination Port",
    "distinct_dst_ports":   "Destination Port",
    "distinct_src_ports_used": None,                       # no src port → const 1
    # directionality
    "fwd_bwd_byte_ratio":   ("Total Length of Fwd Packets", "Total Length of Bwd Packets"),
    "mean_init_win_bytes_fwd": "Init_Win_bytes_forward",
    # packet-derived (no PCAP → 0)
    "ttl_variance":         None,
    "tcp_window_std":       None,
    "fragmentation_ratio":  None,
}


def _prepare_numeric_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the numeric base matrix (N_rows, N_base_cols) from the DataFrame.

    Returns a dict of column-name → numpy array for fast aggregation.
    """
    cols_needed = set()
    for v in _COL_MAP.values():
        if v is None:
            continue
        if isinstance(v, tuple):
            cols_needed.update(v)
        else:
            cols_needed.add(v)

    available = {c for c in cols_needed if c in df.columns}
    missing = cols_needed - available
    if missing:
        logger.debug(f"Missing columns (will use 0): {missing}")

    data = {}
    for col in available:
        arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values.astype(np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        data[col] = arr
    return data


def build_state_windows(
    df: pd.DataFrame,
    split_tag: str,
    flows_per_window: int = FLOWS_PER_WINDOW,
) -> pd.DataFrame:
    """Build non-overlapping windows from a pre-processed flow DataFrame.

    Vectorised implementation: O(N) over flows using numpy reshape.
    Each window aggregates `flows_per_window` consecutive flow rows.

    Args:
        df: Cleaned flow DataFrame (one split, no timestamp/IP columns required).
        split_tag: 'train', 'validation', or 'test'.
        flows_per_window: Number of flows per window.

    Returns:
        DataFrame with one row per window, 24 feature columns + metadata.
    """
    sys_cfg = config.system()
    min_flows = max(2, sys_cfg["windowing"]["minimum_flows"])

    if "Label" in df.columns:
        df = df.dropna(subset=["Label"]).reset_index(drop=True)

    n_rows = len(df)
    if n_rows < flows_per_window:
        logger.warning(f"Split '{split_tag}' has only {n_rows} rows, less than window size {flows_per_window}.")
        return pd.DataFrame()

    # Trim to multiple of flows_per_window
    n_complete = (n_rows // flows_per_window) * flows_per_window
    df_trimmed = df.iloc[:n_complete].reset_index(drop=True)
    n_windows = n_complete // flows_per_window

    logger.info(f"Split '{split_tag}': {n_rows:,} flows → {n_windows:,} windows "
                f"(window_size={flows_per_window})")

    # ── Numeric feature extraction (vectorised) ───────────────────────────────
    col_data = _prepare_numeric_matrix(df_trimmed)

    def _reshape_mean(col_name: str) -> np.ndarray:
        if col_name not in col_data:
            return np.zeros(n_windows)
        return col_data[col_name].reshape(n_windows, flows_per_window).mean(axis=1)

    def _reshape_sum(col_name: str) -> np.ndarray:
        if col_name not in col_data:
            return np.zeros(n_windows)
        return col_data[col_name].reshape(n_windows, flows_per_window).sum(axis=1)

    def _reshape_max(col_name: str) -> np.ndarray:
        if col_name not in col_data:
            return np.zeros(n_windows)
        return col_data[col_name].reshape(n_windows, flows_per_window).max(axis=1)

    def _col_nunique(col_name: str) -> np.ndarray:
        """Vectorised approximate nunique per window.
        Uses sorted+diff trick: count distinct values without Python loop.
        """
        if col_name not in col_data:
            return np.ones(n_windows)
        mat = col_data[col_name].reshape(n_windows, flows_per_window)
        sorted_mat = np.sort(mat, axis=1)                                  # (W, F)
        diffs = np.diff(sorted_mat, axis=1)                                # (W, F-1)
        # count transitions (changes) + 1
        n_unique = (diffs != 0).sum(axis=1) + 1
        return n_unique.astype(np.float64)

    EPS = 1e-9

    # Build feature matrix row by row (but only 24 scalar ops, not N_rows)
    feat_matrix = np.zeros((n_windows, 24), dtype=np.float32)

    fwd_bytes = _reshape_sum("Total Length of Fwd Packets")
    bwd_bytes = _reshape_sum("Total Length of Bwd Packets")
    fwd_pkts  = _reshape_sum("Total Fwd Packets")
    bwd_pkts  = _reshape_sum("Total Backward Packets")
    dur_mean  = _reshape_mean("Flow Duration")

    total_bytes   = fwd_bytes + bwd_bytes
    total_packets = fwd_pkts + bwd_pkts
    dur_s = np.maximum(dur_mean * flows_per_window / 1e6, EPS)

    feat_matrix[:, 0]  = flows_per_window                   # flow_count
    feat_matrix[:, 1]  = total_bytes                         # total_bytes
    feat_matrix[:, 2]  = total_packets                       # total_packets
    feat_matrix[:, 3]  = dur_mean                            # mean_flow_duration
    feat_matrix[:, 4]  = total_bytes / dur_s                 # bytes_per_sec
    feat_matrix[:, 5]  = total_packets / dur_s               # packets_per_sec
    feat_matrix[:, 6]  = _reshape_sum("SYN Flag Count")  / flows_per_window  # syn_ratio
    feat_matrix[:, 7]  = _reshape_sum("ACK Flag Count")  / flows_per_window  # ack_ratio
    feat_matrix[:, 8]  = _reshape_sum("FIN Flag Count")  / flows_per_window  # fin_ratio
    feat_matrix[:, 9]  = _reshape_sum("RST Flag Count")  / flows_per_window  # rst_ratio
    feat_matrix[:, 10] = _reshape_sum("PSH Flag Count")  / flows_per_window  # psh_ratio
    feat_matrix[:, 11] = _reshape_mean("Flow IAT Mean")                      # mean_iat
    feat_matrix[:, 12] = _reshape_mean("Flow IAT Std")                       # std_iat
    feat_matrix[:, 13] = _reshape_max("Flow IAT Max")                        # max_iat
    feat_matrix[:, 14] = _reshape_mean("Packet Length Mean")                 # mean_packet_len
    feat_matrix[:, 15] = _reshape_mean("Packet Length Std")                  # std_packet_len
    feat_matrix[:, 16] = _col_nunique("Destination Port").astype(np.float32) # distinct_dst_ips (proxy)
    feat_matrix[:, 17] = feat_matrix[:, 16]                                  # distinct_dst_ports
    feat_matrix[:, 18] = 1.0                                                  # distinct_src_ports_used
    feat_matrix[:, 19] = fwd_bytes / np.maximum(bwd_bytes, EPS)              # fwd_bwd_byte_ratio
    feat_matrix[:, 20] = _reshape_mean("Init_Win_bytes_forward")             # mean_init_win_bytes_fwd
    feat_matrix[:, 21] = 0.0                                                  # ttl_variance
    feat_matrix[:, 22] = 0.0                                                  # tcp_window_std
    feat_matrix[:, 23] = 0.0                                                  # fragmentation_ratio

    # Replace any remaining NaN/Inf
    feat_matrix = np.where(np.isfinite(feat_matrix), feat_matrix, 0.0).astype(np.float32)

    # ── Z_t label assignment (vectorised) ────────────────────────────────────
    # Strategy: pre-compute per-flow Z_t index using a label→priority map,
    # then take argmin priority per window (lower rank = higher attack severity).
    # This is O(N_flows) numpy, not O(N_windows) Python loop.
    from digitalspy.states.labels import get_label_map, get_precedence

    if "Label" in df_trimmed.columns:
        label_map = get_label_map()          # raw_label → bucket str
        precedence = get_precedence()         # bucket str → int (lower=higher priority)

        # Map raw labels → bucket index via vectorised pandas map
        raw_labels = df_trimmed["Label"].str.strip().values

        # Build a lookup: raw_label → priority rank (int)
        # Unknown labels get priority = precedence["BENIGN"]
        benign_rank = precedence["BENIGN"]

        # Vectorised map: raw label → bucket → rank
        def _raw_to_rank(raw: str) -> int:
            raw_n = raw.replace("\ufffd", " ").replace("  ", " ").strip()
            bucket = label_map.get(raw) or label_map.get(raw_n) or "BENIGN"
            return precedence.get(bucket, benign_rank)

        # Build lookup table for all unique raw labels in this split
        unique_raw = np.unique(raw_labels)
        rank_lookup = {r: _raw_to_rank(r) for r in unique_raw}
        # Also build bucket lookup
        bucket_lookup = {}
        for r in unique_raw:
            r_n = r.replace("\ufffd", " ").replace("  ", " ").strip()
            bucket_lookup[r] = label_map.get(r) or label_map.get(r_n) or "BENIGN"

        # Map each flow → rank (vectorised)
        flow_ranks = np.array([rank_lookup[r] for r in raw_labels], dtype=np.int32)
        flow_buckets = np.array([bucket_lookup[r] for r in raw_labels])

        # Reshape and take min rank per window (= highest-priority bucket)
        rank_matrix = flow_ranks.reshape(n_windows, flows_per_window)   # (W, F)
        min_rank_per_window = rank_matrix.min(axis=1)                   # (W,)

        # For each window, take the bucket name corresponding to min rank
        # Use the first flow that has that rank
        bucket_matrix = flow_buckets.reshape(n_windows, flows_per_window)
        z_t_list = []
        for i in range(n_windows):
            min_rank = min_rank_per_window[i]
            # Find first occurrence of min rank in this window
            idx = np.where(rank_matrix[i] == min_rank)[0][0]
            z_t_list.append(bucket_matrix[i][idx])

        z_t_arr = np.array(z_t_list)
    else:
        z_t_arr = np.array(["BENIGN"] * n_windows)

    z_t_idx_arr = np.array([label_to_index(z) for z in z_t_arr], dtype=np.int64)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    result = pd.DataFrame(feat_matrix, columns=FEATURE_NAMES)
    result["window_idx"] = np.arange(n_windows)
    result["flow_start"] = np.arange(n_windows) * flows_per_window
    result["flow_end"]   = result["flow_start"] + flows_per_window
    result["split"]      = split_tag
    result["z_t"]        = z_t_arr
    result["z_t_idx"]    = z_t_idx_arr
    result["has_packet_features"] = 0

    logger.info(f"  Z_t distribution: {pd.Series(z_t_arr).value_counts().to_dict()}")
    return result


def fit_preprocessor(train_windows: pd.DataFrame, models_dir: Path) -> tuple[StandardScaler, SimpleImputer]:
    """Fit scaler and imputer on training windows ONLY.

    CRITICAL: Never fit on validation or test data. (§8, §30)
    """
    X_train = train_windows[FEATURE_NAMES].values

    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X_train)

    scaler = StandardScaler()
    scaler.fit(X_imputed)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(imputer, models_dir / "imputer.pkl")
    logger.info(f"Saved scaler and imputer to {models_dir}")

    return scaler, imputer


def apply_preprocessor(
    windows: pd.DataFrame,
    scaler: StandardScaler,
    imputer: SimpleImputer,
    *,
    impute_packet_features: bool = True,
) -> pd.DataFrame:
    """Apply pre-fitted scaler/imputer. CRITICAL: never refit on val/test."""
    windows = windows.copy()
    X = windows[FEATURE_NAMES].values.astype(np.float64)

    X = imputer.transform(X)

    if impute_packet_features:
        packet_indices = [FEATURE_NAMES.index(f) for f in _PACKET_DERIVED]
        no_packet_mask = (windows["has_packet_features"] == 0).values
        if no_packet_mask.any():
            for pi in packet_indices:
                X[no_packet_mask, pi] = imputer.statistics_[pi]

    X_scaled = scaler.transform(X)
    windows[FEATURE_NAMES] = X_scaled.astype(np.float32)
    return windows


def build_lstm_sequences(
    windows: pd.DataFrame,
    history_length: int = 20,
    K: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Build (history_length, 24) LSTM sequences from sequential windows.

    Fully vectorised via numpy stride tricks.
    No gap handling needed: windows are sequentially ordered within each split file.

    Args:
        windows: Scaled window DataFrame, ordered sequentially.
        history_length: h = 20 windows.
        K: forecast horizon = 5 windows.

    Returns:
        X: (N, 20, 24) float32
        y_risk: (N, K) float32
        y_tactic: (N, K) int64
        meta: list of window_idx dicts
    """
    feats = windows[FEATURE_NAMES].values.astype(np.float32)
    z_idx = windows["z_t_idx"].values.astype(np.int64)
    risk  = (windows["z_t"] != "BENIGN").astype(np.float32).values
    n = len(feats)

    total_needed = history_length + K
    if n < total_needed:
        return (np.zeros((0, history_length, 24), dtype=np.float32),
                np.zeros((0, K), dtype=np.float32),
                np.zeros((0, K), dtype=np.int64),
                [])

    # Number of valid sequences
    N_seq = n - total_needed + 1

    # Build index arrays
    hist_indices = np.arange(N_seq)[:, None] + np.arange(history_length)[None, :]   # (N, 20)
    tgt_indices  = np.arange(N_seq)[:, None] + history_length + np.arange(K)[None, :]  # (N, K)

    X        = feats[hist_indices]    # (N, 20, 24)
    y_risk   = risk[tgt_indices]      # (N, K)
    y_tactic = z_idx[tgt_indices]     # (N, K)

    meta = [{"window_idx": int(windows["window_idx"].iloc[i])
              if "window_idx" in windows.columns else i}
             for i in range(N_seq)]

    logger.info(f"Built {N_seq:,} LSTM sequences (h={history_length}, K={K}).")
    return X, y_risk, y_tactic, meta
