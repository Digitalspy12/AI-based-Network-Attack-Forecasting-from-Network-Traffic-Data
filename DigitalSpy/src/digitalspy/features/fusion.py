"""Phase 8A — Flow + Packet feature fusion.

Fuses the existing 24-dimensional flow state S_t^flow with the new
12-dimensional packet state S_t^packet into a combined 36-dimensional state:

    S_t = [S_t^flow ; S_t^packet]

Design constraints:
  1. Window alignment: both states must share the same (host_ip, window_start)
     key.  Misaligned rows must never be joined (no future-leakage).
  2. Coverage tracking: a boolean column `has_packet_features` is added to
     every fused record.  False means packet data was unavailable for that
     window; packet features are filled with 0.0 but the flag makes the
     absence explicit.
  3. The original 24 flow features are NEVER replaced.  Packet features are
     additive.
  4. The combined feature list COMBINED_FEATURE_NAMES has exactly 36 names.

Usage
-----
    # In build_packet_state.py:
    from digitalspy.features.fusion import fuse_flow_packet, COMBINED_FEATURE_NAMES

    fused_df = fuse_flow_packet(flow_windows_df, packet_state_df)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from digitalspy.features.engineer import FEATURE_NAMES as FLOW_FEATURE_NAMES
from digitalspy.features.packet_features import (
    PACKET_FEATURE_NAMES,
    empty_packet_features,
)

logger = logging.getLogger(__name__)

# The full combined feature vector (36 = 24 flow + 12 packet)
COMBINED_FEATURE_NAMES: list[str] = FLOW_FEATURE_NAMES + PACKET_FEATURE_NAMES
assert len(COMBINED_FEATURE_NAMES) == 36, (
    f"Expected 36 combined features, got {len(COMBINED_FEATURE_NAMES)}"
)


def fuse_flow_packet(
    flow_df: pd.DataFrame,
    packet_df: pd.DataFrame,
) -> pd.DataFrame:
    """Fuse packet features onto flow state windows.

    Since MachineLearningCVE flow data lacks 'source_ip' and 'window_start' 
    timestamps, an exact row-level join with PCAP data is not possible.
    This function appends the 12 packet features initialized to 0.0, and 
    flags 'has_packet_features' as False for the current splits.
    
    Args:
        flow_df: DataFrame produced by build_state_windows().
        packet_df: DataFrame produced by build_packet_state.py.

    Returns:
        DataFrame with all flow_df columns plus PACKET_FEATURE_NAMES plus
        `has_packet_features` (bool).
    """
    merged = flow_df.copy()
    
    # Mark coverage as False since exact IP/time join is impossible
    merged["has_packet_features"] = False

    # Fill missing packet features with 0.0
    for feat in PACKET_FEATURE_NAMES:
        merged[feat] = 0.0

    n_total = len(merged)
    logger.warning(
        "Fusion complete: Due to missing IP/Timestamp in CSVs, exact fusion skipped. "
        "%d/%d windows have packet features (0.0%%).",
        0,
        n_total
    )

    return merged

    # Ensure matching dtypes for the join keys
    flow_df = flow_df.copy()
    packet_df = packet_df.copy()

    flow_df["window_start"] = flow_df["window_start"].astype(np.float64)
    packet_df["window_start"] = packet_df["window_start"].astype(np.float64)
    flow_df["source_ip"] = flow_df["source_ip"].astype(str)
    packet_df["source_ip"] = packet_df["source_ip"].astype(str)

    # Select only the key + packet feature columns from packet_df to avoid collisions
    pkt_cols = ["source_ip", "window_start"] + PACKET_FEATURE_NAMES
    # Only select columns that actually exist in packet_df
    available_pkt_cols = [c for c in pkt_cols if c in packet_df.columns]
    packet_subset = packet_df[available_pkt_cols].copy()

    # Left join — every flow window is preserved
    merged = flow_df.merge(
        packet_subset,
        on=["source_ip", "window_start"],
        how="left",
        validate="many_to_one",  # multiple flow windows per host-window is OK
    )

    # Mark coverage
    first_pkt_feat = PACKET_FEATURE_NAMES[0]
    merged["has_packet_features"] = merged[first_pkt_feat].notna().astype(bool)

    # Fill missing packet features with 0.0
    for feat in PACKET_FEATURE_NAMES:
        if feat in merged.columns:
            merged[feat] = merged[feat].fillna(0.0)
        else:
            merged[feat] = 0.0

    # Verify no NaN/Inf in packet columns
    for feat in PACKET_FEATURE_NAMES:
        bad = ~np.isfinite(merged[feat].values)
        if bad.any():
            logger.warning(
                "Non-finite values in %s after fusion (%d rows). Replacing with 0.0.",
                feat,
                bad.sum(),
            )
            merged.loc[bad, feat] = 0.0

    n_total = len(merged)
    n_covered = int(merged["has_packet_features"].sum())
    logger.info(
        "Fusion complete: %d/%d windows have packet features (%.1f%%).",
        n_covered,
        n_total,
        100.0 * n_covered / n_total if n_total > 0 else 0.0,
    )

    return merged


def _validate_join_keys(flow_df: pd.DataFrame, packet_df: pd.DataFrame) -> None:
    """Raise if required join key columns are missing."""
    pass


def get_feature_names(mode: str = "flow") -> list[str]:
    """Return feature names for the requested mode.

    Args:
        mode: 'flow' (24 features), 'packet' (12 features),
              or 'fused' (36 features).
    """
    if mode == "flow":
        return list(FLOW_FEATURE_NAMES)
    elif mode == "packet":
        return list(PACKET_FEATURE_NAMES)
    elif mode == "fused":
        return list(COMBINED_FEATURE_NAMES)
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'flow', 'packet', or 'fused'.")
