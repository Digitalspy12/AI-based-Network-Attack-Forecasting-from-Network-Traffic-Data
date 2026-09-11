"""Feature engineering: compute 24-dimensional S_t from raw flow records.

CIC-IDS2017 flows are pre-computed (Argus/CICFlowMeter).
This module aggregates per-flow features into per-host 10-second window
summary statistics that form the continuous state vector S_t ∈ ℝ²⁴.

Frozen feature list (must be exactly 24):
  Volume/timing (6): flow_count, total_bytes, total_packets,
                     mean_flow_duration, bytes_per_sec, packets_per_sec
  TCP flag ratios (5): syn_ratio, ack_ratio, fin_ratio, rst_ratio, psh_ratio
  Inter-arrival (3): mean_iat, std_iat, max_iat
  Packet size (2): mean_packet_len, std_packet_len
  Diversity (3): distinct_dst_ips, distinct_dst_ports, distinct_src_ports_used
  Directionality (2): fwd_bwd_byte_ratio, mean_init_win_bytes_fwd
  Packet-derived (3): ttl_variance, tcp_window_std, fragmentation_ratio
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from digitalspy import config

logger = logging.getLogger(__name__)

FEATURE_NAMES: list[str] = config.features()["feature_names"]
assert len(FEATURE_NAMES) == 24, f"Expected 24 features, got {len(FEATURE_NAMES)}"

# Small epsilon to avoid division by zero
_EPS = 1e-9


def engineer_window_features(window_df: pd.DataFrame) -> dict[str, float]:
    """Compute S_t for a single host-window DataFrame.

    Args:
        window_df: Subset of flow records for one source IP in one 10s window.
                   Must have columns from CIC-IDS2017 (already cleaned).

    Returns:
        dict mapping each of the 24 feature names to a float value.
        All NaN/Inf are resolved within this function.
    """
    n = len(window_df)
    feats: dict[str, float] = {}

    # ── Volume / timing ──────────────────────────────────────────────────────
    feats["flow_count"] = float(n)

    total_fwd = _safe_sum(window_df, "Total Length of Fwd Packets")
    total_bwd = _safe_sum(window_df, "Total Length of Bwd Packets")
    feats["total_bytes"] = total_fwd + total_bwd

    total_fwd_pkts = _safe_sum(window_df, "Total Fwd Packets")
    total_bwd_pkts = _safe_sum(window_df, "Total Backward Packets")
    feats["total_packets"] = total_fwd_pkts + total_bwd_pkts

    feats["mean_flow_duration"] = _safe_mean(window_df, "Flow Duration")

    total_dur_s = feats["mean_flow_duration"] * n / 1e6  # μs → s
    feats["bytes_per_sec"] = feats["total_bytes"] / max(total_dur_s, _EPS)
    feats["packets_per_sec"] = feats["total_packets"] / max(total_dur_s, _EPS)

    # ── TCP flag ratios ───────────────────────────────────────────────────────
    feats["syn_ratio"] = _flag_ratio(window_df, "SYN Flag Count", n)
    feats["ack_ratio"] = _flag_ratio(window_df, "ACK Flag Count", n)
    feats["fin_ratio"] = _flag_ratio(window_df, "FIN Flag Count", n)
    feats["rst_ratio"] = _flag_ratio(window_df, "RST Flag Count", n)
    feats["psh_ratio"] = _flag_ratio(window_df, "PSH Flag Count", n)

    # ── Inter-arrival ─────────────────────────────────────────────────────────
    feats["mean_iat"] = _safe_mean(window_df, "Flow IAT Mean")
    feats["std_iat"] = _safe_mean(window_df, "Flow IAT Std")
    feats["max_iat"] = _safe_max(window_df, "Flow IAT Max")

    # ── Packet size ───────────────────────────────────────────────────────────
    total_pkts = max(feats["total_packets"], _EPS)
    feats["mean_packet_len"] = feats["total_bytes"] / total_pkts
    feats["std_packet_len"] = _approx_std_packet_len(window_df)

    # ── Diversity ─────────────────────────────────────────────────────────────
    feats["distinct_dst_ips"] = float(
        window_df["Destination IP"].nunique() if "Destination IP" in window_df.columns else 0
    )
    feats["distinct_dst_ports"] = float(
        window_df["Destination Port"].nunique() if "Destination Port" in window_df.columns else 0
    )
    feats["distinct_src_ports_used"] = float(
        window_df["Source Port"].nunique() if "Source Port" in window_df.columns else 0
    )

    # ── Directionality ────────────────────────────────────────────────────────
    feats["fwd_bwd_byte_ratio"] = total_fwd / max(total_bwd, _EPS)
    feats["mean_init_win_bytes_fwd"] = _safe_mean(window_df, "Init_Win_bytes_forward")

    # ── Packet-derived features (PCAP-sourced; imputed if unavailable) ────────
    # These are 0.0 here; imputation happens in the windowing pipeline
    # when has_packet_features=0
    feats["ttl_variance"] = 0.0
    feats["tcp_window_std"] = 0.0
    feats["fragmentation_ratio"] = 0.0

    # Resolve any remaining NaN/Inf
    for k in feats:
        v = feats[k]
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats


def _safe_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


def _safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) > 0 else 0.0


def _safe_max(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.max()) if len(vals) > 0 else 0.0


def _flag_ratio(df: pd.DataFrame, col: str, n: int) -> float:
    if n == 0 or col not in df.columns:
        return 0.0
    total = pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum()
    return float(total) / float(n)


def _approx_std_packet_len(df: pd.DataFrame) -> float:
    """Approximate std of packet lengths using fwd/bwd means and stds."""
    fwd_std = _safe_mean(df, "Fwd Packet Length Std")
    bwd_std = _safe_mean(df, "Bwd Packet Length Std")
    return (fwd_std + bwd_std) / 2.0


def validate_feature_vector(feats: dict[str, float]) -> None:
    """Raise if the feature dict is not exactly 24 features with correct names."""
    actual_keys = set(feats.keys())
    expected_keys = set(FEATURE_NAMES)
    # Exclude the optional packet-derived keys from validation here
    # (they will be imputed later)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        raise ValueError(f"Feature vector missing: {missing}")
    if extra:
        raise ValueError(f"Feature vector has unexpected keys: {extra}")
    if len(feats) != 24:
        raise ValueError(f"Expected exactly 24 features, got {len(feats)}")
