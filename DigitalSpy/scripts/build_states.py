#!/usr/bin/env python3
"""Phase 2 — Build State Windows.

Converts raw CIC-IDS2017 CSVs into the temporal state dataset:
    data/processed/state_windows.parquet

Pipeline:
    1. Load each split (train / validation / test)
    2. Compute 24-feature S_t per 10-second host window
    3. Compute Z_t from raw labels via precedence (never from S_t values)
    4. Fit scaler + imputer on TRAIN ONLY
    5. Apply preprocessing to all splits
    6. Build LSTM sequences (20-window, K=5, discard on gap)
    7. Save state_windows.parquet, *_sequences.npz files

CRITICAL: scaler/imputer fitted on training data only. (§8, §30)
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.data.loader import load_split
from digitalspy.states.windowing import (
    build_state_windows,
    fit_preprocessor,
    apply_preprocessor,
    build_lstm_sequences,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    sys_cfg = config.system()
    raw_dir = config.resolve_path("raw_data")
    processed_dir = config.resolve_path("processed_data")
    models_dir = config.resolve_path("models")
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    lstm_cfg = config.lstm()
    h = sys_cfg["windowing"]["history_length"]
    K = sys_cfg["forecast"]["K"]

    # ── Phase 2a: Load raw data ─────────────────────────────────────────────
    logger.info("Loading splits…")
    train_raw = load_split("train", raw_dir)
    val_raw = load_split("validation", raw_dir)
    test_raw = load_split("test", raw_dir)

    # ── Phase 2b: Build windows ─────────────────────────────────────────────
    logger.info("Building host-windows…")
    train_windows = build_state_windows(train_raw, split_tag="train")
    val_windows = build_state_windows(val_raw, split_tag="validation")
    test_windows = build_state_windows(test_raw, split_tag="test")

    if len(train_windows) == 0:
        logger.error("No training windows generated. Check raw data and timestamp parsing.")
        sys.exit(1)

    # ── Phase 2c: Fit preprocessor on TRAIN ONLY ───────────────────────────
    logger.info("Fitting scaler/imputer on training data only…")
    scaler, imputer = fit_preprocessor(train_windows, models_dir)

    # ── Phase 2d: Apply to all splits ──────────────────────────────────────
    logger.info("Applying preprocessing…")
    train_scaled = apply_preprocessor(train_windows, scaler, imputer)
    val_scaled = apply_preprocessor(val_windows, scaler, imputer)
    test_scaled = apply_preprocessor(test_windows, scaler, imputer)

    # ── Phase 2e: Save combined state windows ──────────────────────────────
    all_windows = pd.concat([train_scaled, val_scaled, test_scaled], ignore_index=True)
    state_path = processed_dir / "state_windows.parquet"
    all_windows.to_parquet(state_path, index=False)
    logger.info(f"✓ Saved state_windows.parquet: {len(all_windows):,} rows → {state_path}")

    # ── Phase 2f: Build LSTM sequences ─────────────────────────────────────
    logger.info("Building LSTM sequences (history=20, K=5, discard-on-gap)…")

    def save_sequences(windows, tag):
        X, y_risk, y_tactic, meta = build_lstm_sequences(windows, h, K)
        out_path = processed_dir / f"{tag}_sequences.npz"
        np.savez_compressed(out_path, X=X, y_risk=y_risk, y_tactic=y_tactic)
        logger.info(f"  {tag}: {len(X):,} sequences → {out_path}")
        return X, y_risk, y_tactic

    X_tr, yr_tr, yt_tr = save_sequences(train_scaled, "train")
    X_va, yr_va, yt_va = save_sequences(val_scaled, "val")
    X_te, yr_te, yt_te = save_sequences(test_scaled, "test")

    # ── Summary ─────────────────────────────────────────────────────────────
    summary = {
        "train_windows": int(len(train_scaled)),
        "val_windows": int(len(val_scaled)),
        "test_windows": int(len(test_scaled)),
        "train_sequences": int(len(X_tr)),
        "val_sequences": int(len(X_va)),
        "test_sequences": int(len(X_te)),
        "feature_dim": 24,
        "history_length": h,
        "K": K,
        "z_t_distribution_train": train_scaled["z_t"].value_counts().to_dict(),
        "z_t_distribution_val": val_scaled["z_t"].value_counts().to_dict(),
        "z_t_distribution_test": test_scaled["z_t"].value_counts().to_dict(),
    }

    reports_dir = config.resolve_path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "state_pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("STATE PIPELINE SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n✓ Phase 2 complete. Proceed to Phase 3 (train_logistic.py).")


if __name__ == "__main__":
    main()
