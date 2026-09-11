"""Label precedence engine for Z_t computation.

Z_t is derived from raw CIC-IDS2017 labels via the project's precedence rule.
It is NEVER computed from the numerical values of S_t.

Precedence (lowest rank = highest priority):
    IMPACT (1) > C2 (2) > LATERAL_MOVEMENT (3) > INITIAL_ACCESS (4) > RECON (5) > BENIGN (6)

BENIGN applies ONLY when no attack label is present in the window.
"""
from __future__ import annotations

from typing import Iterable

from digitalspy import config


def get_label_map() -> dict[str, str]:
    """Return raw label → bucket mapping from config."""
    return config.labels()["label_map"]


def get_precedence() -> dict[str, int]:
    """Return bucket → rank mapping (lower = higher priority)."""
    return config.labels()["precedence"]


def map_raw_label(raw_label: str) -> str:
    """Map a single raw CIC-IDS2017 label string to a Z_t bucket.

    Args:
        raw_label: Stripped raw label string from the dataset.

    Returns:
        Z_t bucket name string.

    Raises:
        ValueError: If the label is not in the label map.
    """
    label_map = get_label_map()
    raw_label = raw_label.strip()

    if raw_label in label_map:
        return label_map[raw_label]

    # Attempt partial match for Web Attack variants
    for key, bucket in label_map.items():
        if raw_label.lower() == key.lower():
            return bucket

    raise ValueError(f"Unknown raw label: '{raw_label}'. "
                     f"Known labels: {list(label_map.keys())}")


def resolve_window_label(raw_labels: Iterable[str]) -> str:
    """Apply label precedence to determine Z_t for a window.

    Algorithm:
        tactic(window) = min(rank(label) for label in labels_present)

    BENIGN only when no attack label is in the window.

    Args:
        raw_labels: Iterable of raw label strings in the window.

    Returns:
        The Z_t bucket with the highest precedence (lowest rank).
    """
    precedence = get_precedence()
    label_map = get_label_map()

    buckets_in_window = set()
    for raw in raw_labels:
        raw = str(raw).strip()
        # CIC-IDS2017 uses \ufffd (unicode replacement) in "Web Attack ´ Brute Force"
        # Normalize to a simple space for matching
        raw_normalized = raw.replace("\ufffd", " ").replace("  ", " ").strip()
        matched = False
        if raw in label_map:
            buckets_in_window.add(label_map[raw])
            matched = True
        elif raw_normalized in label_map:
            buckets_in_window.add(label_map[raw_normalized])
            matched = True
        else:
            # Try case-insensitive
            for key, bucket in label_map.items():
                if raw.lower() == key.lower() or raw_normalized.lower() == key.replace("\ufffd", " ").lower():
                    buckets_in_window.add(bucket)
                    matched = True
                    break
        if not matched:
            # Treat unknown labels as BENIGN with a warning (logged upstream)
            buckets_in_window.add("BENIGN")

    if not buckets_in_window:
        return "BENIGN"

    # Select the bucket with the minimum rank (= highest priority)
    return min(buckets_in_window, key=lambda b: precedence.get(b, 999))


def label_to_index(label: str) -> int:
    """Convert Z_t bucket name to integer index."""
    indices = config.labels()["state_indices"]
    if label not in indices:
        raise ValueError(f"Unknown Z_t label: '{label}'")
    return indices[label]


def index_to_label(idx: int) -> str:
    """Convert integer index to Z_t bucket name."""
    indices = config.labels()["state_indices"]
    inv = {v: k for k, v in indices.items()}
    if idx not in inv:
        raise ValueError(f"Unknown Z_t index: {idx}")
    return inv[idx]


def is_attack(label: str) -> bool:
    """Return True if the Z_t label is any attack state (not BENIGN)."""
    return label != "BENIGN"
