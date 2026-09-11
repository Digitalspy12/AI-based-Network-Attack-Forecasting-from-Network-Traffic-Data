"""CIC-IDS2017 data loader and validator.

Handles the known quirks of CIC-IDS2017:
- Column names with leading/trailing whitespace
- Infinity values in numeric columns
- Label column whitespace
- Mixed float/int types
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from digitalspy import config

logger = logging.getLogger(__name__)

# CIC-IDS2017 known numeric columns that can contain Inf
_PROBLEMATIC_COLS = [
    "Flow Bytes/s",
    "Flow Packets/s",
]


def load_csv(path: Path, *, validate: bool = True) -> pd.DataFrame:
    """Load a single CIC-IDS2017 CSV with cleaning.

    Steps:
    1. Strip whitespace from all column names.
    2. Strip whitespace from Label column.
    3. Replace ±Infinity with NaN.
    4. Optionally validate schema.

    Args:
        path: Absolute path to CSV file.
        validate: If True, warn on unexpected columns.

    Returns:
        Cleaned DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    file_size_mb = path.stat().st_size / (1024 ** 2)
    logger.info(f"Loading {path.name} ({file_size_mb:.1f} MB)…")

    df = pd.read_csv(path, low_memory=False)

    # Strip column name whitespace (CIC-IDS2017 known issue)
    df.columns = df.columns.str.strip()

    # Strip label whitespace
    if "Label" in df.columns:
        df["Label"] = df["Label"].str.strip()

    # Replace ±Inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    logger.info(f"  → {len(df):,} rows, {df.shape[1]} columns, "
                f"{df['Label'].value_counts().to_dict() if 'Label' in df.columns else 'no Label'}")

    if validate:
        _validate_schema(df, path.name)

    return df


def load_split(split_name: str, raw_data_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load all CSV files for a given split (train / validation / test).

    Args:
        split_name: One of 'train', 'validation', 'test'.
        raw_data_dir: Override path to raw data directory.

    Returns:
        Concatenated DataFrame for the split, sorted by Timestamp.
    """
    splits_cfg = config.splits()
    if split_name not in splits_cfg["splits"]:
        raise ValueError(f"Unknown split: {split_name}. Must be one of {list(splits_cfg['splits'].keys())}")

    if raw_data_dir is None:
        raw_data_dir = config.resolve_path("raw_data")

    file_list = splits_cfg["splits"][split_name]["files"]
    dfs = []

    for fname in file_list:
        fpath = raw_data_dir / fname
        df = load_csv(fpath)
        df["_source_file"] = fname
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Parse and sort by timestamp
    if "Timestamp" in combined.columns:
        combined["Timestamp"] = pd.to_datetime(combined["Timestamp"], dayfirst=True, errors="coerce")
        combined.sort_values("Timestamp", inplace=True)
        combined.reset_index(drop=True, inplace=True)

    logger.info(f"Split '{split_name}': {len(combined):,} rows from {len(file_list)} files.")
    return combined


def _validate_schema(df: pd.DataFrame, filename: str) -> None:
    """Warn if expected columns are missing."""
    expected = {"Label", "Source IP", "Destination IP",
                "Source Port", "Destination Port", "Timestamp"}
    missing = expected - set(df.columns)
    if missing:
        logger.warning(f"{filename}: Missing expected columns: {missing}")
