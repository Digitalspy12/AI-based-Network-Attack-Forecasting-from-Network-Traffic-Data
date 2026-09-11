#!/usr/bin/env python3
"""Forecast Lead Time Evaluation (Criterion 3).

Algorithm (IMPLEMENTATION.md §18):
    1. Alert threshold chosen on VALIDATION data only — frozen before test.
    2. For each test attack:
           LeadTime = AttackOnset - FirstForecast
    3. Report: median, mean, positive-lead-time rate.

CRITICAL: Threshold is chosen on validation, NOT test data. (§18, §29)
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.models.attention_lstm import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

THRESHOLD_CANDIDATES = np.arange(0.3, 0.9, 0.05)


def choose_threshold_on_validation(
    model,
    X_val: np.ndarray,
    y_risk_val: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Select alert threshold using validation data only.

    Strategy: maximise F1 on validation set for k=1 risk.
    Freeze before test evaluation.
    """
    from sklearn.metrics import f1_score

    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            xb = torch.FloatTensor(X_val[i:i+batch_size]).to(device)
            risk_probs, _, _ = model(xb)
            probs.append(risk_probs[:, 0].cpu().numpy())

    probs = np.concatenate(probs)
    y_true = y_risk_val[:, 0]

    best_f1, best_thresh = 0.0, 0.5
    for t in THRESHOLD_CANDIDATES:
        preds = (probs > t).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(t)

    logger.info(f"Threshold selected on VALIDATION: {best_thresh:.2f} (val F1={best_f1:.4f})")
    return best_thresh


def compute_lead_times(
    model,
    X_test: np.ndarray,
    y_risk_test: np.ndarray,
    threshold: float,
    window_seconds: int,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Compute per-sequence forecast lead times.

    For each test sequence where an attack occurs in the forecast horizon:
        LeadTime = (first horizon k where ŷ_risk > threshold) - actual_onset_k
    """
    model.eval()
    K = y_risk_test.shape[1]

    all_risk_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            xb = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            risk_probs, _, _ = model(xb)
            all_risk_probs.append(risk_probs.cpu().numpy())

    risk_probs = np.vstack(all_risk_probs)  # (N, K)

    lead_times = []
    for i in range(len(y_risk_test)):
        # Find actual attack onset (first k where label is attack)
        attack_onsets = np.where(y_risk_test[i] == 1)[0]
        if len(attack_onsets) == 0:
            continue  # BENIGN sequence, skip

        attack_k = int(attack_onsets[0])
        attack_onset_sec = (attack_k + 1) * window_seconds

        # Find first qualifying forecast (risk > threshold) at any horizon
        forecast_ks = np.where(risk_probs[i] > threshold)[0]
        if len(forecast_ks) == 0:
            # No forecast triggered — negative lead time (missed)
            lead_times.append({
                "attack_onset_k": attack_k,
                "attack_onset_sec": attack_onset_sec,
                "first_forecast_k": None,
                "first_forecast_sec": None,
                "lead_time_sec": -(attack_onset_sec),  # -ve = missed
                "detected": False,
            })
            continue

        first_forecast_k = int(forecast_ks[0])
        first_forecast_sec = (first_forecast_k + 1) * window_seconds

        lead_time = attack_onset_sec - first_forecast_sec
        lead_times.append({
            "attack_onset_k": attack_k,
            "attack_onset_sec": attack_onset_sec,
            "first_forecast_k": first_forecast_k,
            "first_forecast_sec": first_forecast_sec,
            "lead_time_sec": float(lead_time),
            "detected": True,
        })

    return lead_times


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir = config.resolve_path("models")
    reports_dir = config.resolve_path("reports")
    lstm_cfg = config.lstm()
    sys_cfg = config.system()
    window_sec = sys_cfg["windowing"]["window_seconds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load sequences
    def load_seq(tag):
        d = np.load(processed_dir / f"{tag}_sequences.npz")
        return d["X"], d["y_risk"], d["y_tactic"]

    X_va, yr_va, _ = load_seq("val")
    X_te, yr_te, _ = load_seq("test")

    # Load model
    model = build_model(lstm_cfg).to(device)
    checkpoint_path = models_dir / "lstm_checkpoint.pt"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # ── 1. Choose threshold on VALIDATION (frozen) ──────────────────────────
    threshold = choose_threshold_on_validation(model, X_va, yr_va, device)

    # ── 2. Compute lead times on TEST ────────────────────────────────────────
    logger.info("Computing lead times on test set…")
    lead_times = compute_lead_times(model, X_te, yr_te, threshold, window_sec, device)

    if not lead_times:
        logger.warning("No attack sequences found in test set.")
        return

    lt_values = [l["lead_time_sec"] for l in lead_times]
    positive_count = sum(1 for l in lead_times if l["lead_time_sec"] > 0)
    total_attacks = len(lead_times)

    median_lt = float(np.median(lt_values))
    mean_lt = float(np.mean(lt_values))
    positive_rate = positive_count / total_attacks if total_attacks > 0 else 0.0

    # ── 3. Criterion 3 check ─────────────────────────────────────────────────
    criterion_3_passes = bool(median_lt >= 10.0 and positive_rate >= 0.60)

    report = {
        "threshold_used": round(threshold, 3),
        "threshold_selected_on": "validation",
        "total_attack_sequences": total_attacks,
        "median_lead_time_sec": round(median_lt, 2),
        "mean_lead_time_sec": round(mean_lt, 2),
        "positive_lead_time_count": positive_count,
        "positive_lead_time_rate": round(positive_rate, 4),
        "criterion_3": {
            "median_lead_time_target": 10.0,
            "positive_rate_target": 0.60,
            "median_passes": bool(median_lt >= 10.0),
            "rate_passes": bool(positive_rate >= 0.60),
            "overall_passes": criterion_3_passes,
        },
        "lead_time_details": lead_times[:50],  # sample
    }

    report_path = reports_dir / "lead_time_metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 60)
    print("FORECAST LEAD TIME (CRITERION 3)")
    print("=" * 60)
    print(f"  Alert threshold (validation):  {threshold:.2f}")
    print(f"  Total attack test sequences:   {total_attacks}")
    print(f"  Median lead time:              {median_lt:.1f}s (target ≥10s)")
    print(f"  Mean lead time:                {mean_lt:.1f}s")
    print(f"  Positive lead time rate:       {positive_rate:.1%} (target ≥60%)")
    print(f"\nCriterion 3: {'PASS ✓' if criterion_3_passes else 'FAIL ✗ (valid negative result)'}")
    if not criterion_3_passes:
        print("  NOTE: A zero/negative result is a legitimate negative research result (§17).")
    print(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    main()
