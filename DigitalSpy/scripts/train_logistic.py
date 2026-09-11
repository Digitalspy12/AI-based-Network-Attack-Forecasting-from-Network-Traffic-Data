#!/usr/bin/env python3
"""Phase 3 — Train Logistic Regression Baselines.

Two experiments per IMPLEMENTATION.md §10 and §9:
    Experiment A: S_t → LR → Y_t   (current detection)
    Experiment B (baseline): S_t → LR → Y_{t+1}  (one-step forecasting)

Outputs:
    models/logistic_current_risk.pkl
    models/logistic_current_tactic.pkl
    models/logistic_onestep_risk.pkl
    models/logistic_onestep_tactic.pkl
    reports/baseline_metrics.json

IMPORTANT: Accuracy is NOT reported as headline metric.
Headline: Precision, Recall, Macro-F1, FPR.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.baselines.logistic import LogisticBaseline
from digitalspy.features.engineer import FEATURE_NAMES
from digitalspy.states.labels import index_to_label

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_windows(split: str, processed_dir: Path) -> pd.DataFrame:
    path = processed_dir / "state_windows.parquet"
    if not path.exists():
        logger.error(f"state_windows.parquet not found. Run build_states.py first.")
        sys.exit(1)
    df = pd.read_parquet(path)
    return df[df["split"] == split].reset_index(drop=True)


def build_onestep_targets(windows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build shifted targets: X[i] → y[i+1].

    Groups by source_ip + window_start if those columns exist; otherwise
    falls back to treating all consecutive rows (sorted by window_idx) as
    a single sequence — compatible with CIC-IDS-2017 which has no IP columns.
    """
    HAS_IP = "source_ip" in windows.columns
    HAS_TS = "window_start" in windows.columns

    if HAS_IP and HAS_TS:
        X_list, y_risk_list, y_tactic_list = [], [], []
        for src_ip, grp in windows.groupby("source_ip"):
            grp = grp.sort_values("window_start").reset_index(drop=True)
            if len(grp) < 2:
                continue
            X = grp[FEATURE_NAMES].values[:-1]
            y_risk = (grp["z_t"].values[1:] != "BENIGN").astype(int)
            y_tactic = grp["z_t_idx"].values[1:]
            X_list.append(X)
            y_risk_list.append(y_risk)
            y_tactic_list.append(y_tactic)
        if not X_list:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        return (
            np.vstack(X_list),
            np.concatenate(y_risk_list),
            np.concatenate(y_tactic_list),
        )
    else:
        # Fallback: no IP/timestamp — use global consecutive-row pairs
        logger.warning(
            "source_ip/window_start columns not found; building one-step targets "
            "from consecutive rows sorted by window_idx."
        )
        sort_col = "window_idx" if "window_idx" in windows.columns else None
        if sort_col:
            windows = windows.sort_values(sort_col).reset_index(drop=True)
        if len(windows) < 2:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        X = windows[FEATURE_NAMES].values[:-1]
        y_risk = (windows["z_t"].values[1:] != "BENIGN").astype(int)
        y_tactic = windows["z_t_idx"].values[1:]
        return X, y_risk, y_tactic


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir = config.resolve_path("models")
    reports_dir = config.resolve_path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    label_cfg = config.labels()
    class_names = label_cfg["states"]

    # ── Load splits ─────────────────────────────────────────────────────────
    train_w = load_windows("train", processed_dir)
    val_w = load_windows("validation", processed_dir)
    test_w = load_windows("test", processed_dir)

    X_tr = train_w[FEATURE_NAMES].values
    X_va = val_w[FEATURE_NAMES].values
    X_te = test_w[FEATURE_NAMES].values

    y_risk_tr = (train_w["z_t"] != "BENIGN").astype(int).values
    y_risk_va = (val_w["z_t"] != "BENIGN").astype(int).values
    y_risk_te = (test_w["z_t"] != "BENIGN").astype(int).values

    y_tactic_tr = train_w["z_t_idx"].values
    y_tactic_va = val_w["z_t_idx"].values
    y_tactic_te = test_w["z_t_idx"].values

    all_metrics = {}

    # ── Experiment A: Current Detection ─────────────────────────────────────
    logger.info("Experiment A: Current detection S_t → Y_t")

    # A1: Binary risk
    lr_curr_risk = LogisticBaseline(task="binary")
    lr_curr_risk.fit(X_tr, y_risk_tr)
    lr_curr_risk.save(models_dir / "logistic_current_risk.pkl")
    m = lr_curr_risk.evaluate(X_te, y_risk_te, label="A1_current_risk_test")
    all_metrics["A1_current_risk_test"] = m
    logger.info(f"  A1 test Macro-F1={m['macro_f1']:.4f}, FPR={m['fpr']:.4f}")

    # A2: 6-class tactic
    lr_curr_tactic = LogisticBaseline(task="multiclass", class_names=class_names)
    lr_curr_tactic.fit(X_tr, y_tactic_tr)
    lr_curr_tactic.save(models_dir / "logistic_current_tactic.pkl")
    m = lr_curr_tactic.evaluate(X_te, y_tactic_te, label="A2_current_tactic_test")
    all_metrics["A2_current_tactic_test"] = m
    logger.info(f"  A2 test Macro-F1={m['macro_f1']:.4f}")

    # ── Experiment B: One-Step Forecasting Baseline ──────────────────────────
    logger.info("Experiment B: One-step forecast S_t → Y_{t+1}")

    X_tr1, y_risk_tr1, y_tactic_tr1 = build_onestep_targets(train_w)
    X_va1, y_risk_va1, y_tactic_va1 = build_onestep_targets(val_w)
    X_te1, y_risk_te1, y_tactic_te1 = build_onestep_targets(test_w)

    # B1: Binary risk one-step
    lr_os_risk = LogisticBaseline(task="binary")
    lr_os_risk.fit(X_tr1, y_risk_tr1)
    lr_os_risk.save(models_dir / "logistic_onestep_risk.pkl")
    m = lr_os_risk.evaluate(X_te1, y_risk_te1, label="B1_onestep_risk_test")
    all_metrics["B1_onestep_risk_test"] = m
    logger.info(f"  B1 test Macro-F1={m['macro_f1']:.4f}, FPR={m['fpr']:.4f}")

    # B2: 6-class tactic one-step
    lr_os_tactic = LogisticBaseline(task="multiclass", class_names=class_names)
    lr_os_tactic.fit(X_tr1, y_tactic_tr1)
    lr_os_tactic.save(models_dir / "logistic_onestep_tactic.pkl")
    m = lr_os_tactic.evaluate(X_te1, y_tactic_te1, label="B2_onestep_tactic_test")
    all_metrics["B2_onestep_tactic_test"] = m
    logger.info(f"  B2 test Macro-F1={m['macro_f1']:.4f}")

    # ── Save report ─────────────────────────────────────────────────────────
    report_path = reports_dir / "baseline_metrics.json"
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("LOGISTIC REGRESSION BASELINE METRICS")
    print("=" * 60)
    for name, metrics in all_metrics.items():
        print(f"\n{name}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            elif isinstance(v, dict):
                print(f"  {k}: (per-class)")
    print(f"\n✓ Phase 3 complete. Metrics saved to {report_path}")
    print(f"  → B1 LR one-step risk Macro-F1 = {all_metrics['B1_onestep_risk_test']['macro_f1']:.4f}")
    print(f"    (LSTM must beat this by ≥5pp to pass Criterion 1)")


if __name__ == "__main__":
    main()
