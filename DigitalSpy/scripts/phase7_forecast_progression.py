#!/usr/bin/env python3
"""Phase 7 — Controlled Forecasting Progression Experiment.

Tests the core DigitalSpy claim:
    "Can the model forecast a future attack state BEFORE it becomes
     observable in the current window?"

Strategy (nextstep.md §Phase 7):
    Use VALIDATION sequences where:
        current state = BENIGN  (y_risk[i, 0] = 0)
        future state  = ATTACK  (y_risk[i, k] = 1 for some k ≥ 2..5)

    These sequences capture the BENIGN → INITIAL_ACCESS transition
    in Thursday's WebAttack scenario — genuine temporal progression.

    Threshold: frozen from Phase 6 (0.90) — NOT re-selected on this data.

Metrics:
    - Proactive detection rate at each horizon k
    - Forecast lead time distribution
    - Median / mean / P10 / P90 lead time
    - Comparison: LSTM proactive vs LSTM reactive

CRITICAL: No test data used. Validation sequences only.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.models.attention_lstm import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

SEP = "=" * 60

# Frozen from Phase 6 validation threshold selection
FROZEN_THRESHOLD = 0.90


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir    = config.resolve_path("models")
    reports_dir   = config.resolve_path("reports")
    sys_cfg       = config.system()
    lstm_cfg      = config.lstm()
    window_sec    = sys_cfg["windowing"]["window_seconds"]
    K             = lstm_cfg["forecast"]["K"]
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model (frozen checkpoint) ──────────────────────────────────────
    model = build_model(lstm_cfg).to(device)
    ckpt = models_dir / "lstm_checkpoint.pt"
    if not ckpt.exists():
        logger.error("lstm_checkpoint.pt not found.")
        sys.exit(1)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    logger.info(f"Loaded LSTM checkpoint. Device: {device}")

    # ── Load validation sequences ────────────────────────────────────────────
    val = np.load(processed_dir / "val_sequences.npz")
    X_val, y_risk_val, y_tactic_val = val["X"], val["y_risk"], val["y_tactic"]
    logger.info(f"Validation sequences: {len(X_val):,}")

    # ── Identify proactive sequences: NOW=BENIGN, FUTURE=ATTACK ─────────────
    current_benign = y_risk_val[:, 0] == 0           # no attack in current window
    future_attack  = y_risk_val[:, 1:].sum(axis=1) > 0  # attack appears in k=2..5
    proactive_mask = current_benign & future_attack
    proactive_idx  = np.where(proactive_mask)[0]

    logger.info(f"Proactive sequences (NOW=BENIGN, FUTURE=attack): {proactive_mask.sum():,}")
    logger.info(f"  Out of total val sequences: {len(X_val):,}")

    if len(proactive_idx) == 0:
        logger.warning("No proactive sequences found. Cannot compute lead time.")
        sys.exit(0)

    X_pro     = X_val[proactive_idx]
    yr_pro    = y_risk_val[proactive_idx]
    yt_pro    = y_tactic_val[proactive_idx]

    # ── Run LSTM inference on proactive sequences ────────────────────────────
    batch_size = 256
    all_risk_probs = []
    with torch.no_grad():
        for i in range(0, len(X_pro), batch_size):
            xb = torch.FloatTensor(X_pro[i:i+batch_size]).to(device)
            risk_probs, _, _ = model(xb)
            all_risk_probs.append(risk_probs.cpu().numpy())
    risk_probs = np.vstack(all_risk_probs)   # (N_pro, K)

    # ── Compute per-sequence metrics ──────────────────────────────────────────
    lead_times = []
    proactive_detections = []    # True if alert fired BEFORE attack onset
    per_horizon_detected = [0] * K   # how many sequences alerted at each k

    for i in range(len(yr_pro)):
        # Find actual attack onset horizon (first k where label=attack)
        onset_ks = np.where(yr_pro[i] == 1)[0]
        if len(onset_ks) == 0:
            continue
        attack_k    = int(onset_ks[0])  # 0-indexed horizon (0=k1, 1=k2, …)
        attack_t    = (attack_k + 1) * window_sec   # seconds into forecast

        # Find first horizon where LSTM raises alert (risk > threshold)
        alert_ks = np.where(risk_probs[i] > FROZEN_THRESHOLD)[0]

        if len(alert_ks) == 0:
            # Missed — no alert raised at any horizon
            lead_times.append(-attack_t)   # negative = missed
            proactive_detections.append(False)
            continue

        alert_k = int(alert_ks[0])
        alert_t = (alert_k + 1) * window_sec
        per_horizon_detected[alert_k] += 1

        lead_time = attack_t - alert_t   # positive = alert before attack
        lead_times.append(lead_time)
        proactive_detections.append(lead_time > 0)

    lead_times = np.array(lead_times)
    positive_lead = np.array(proactive_detections)

    total            = len(lead_times)
    detected         = (lead_times > -999999).sum()   # all (missed = negative)
    proactive_count  = positive_lead.sum()
    proactive_rate   = float(proactive_count / total) if total > 0 else 0.0

    positive_lt      = lead_times[lead_times > 0]
    median_lt        = float(np.median(lead_times))
    mean_lt          = float(np.mean(lead_times))
    median_positive  = float(np.median(positive_lt)) if len(positive_lt) > 0 else 0.0
    p10_lt           = float(np.percentile(lead_times, 10))
    p90_lt           = float(np.percentile(lead_times, 90))

    # Per-horizon F1 on proactive sequences
    horizon_f1s = {}
    for k in range(K):
        preds = (risk_probs[:, k] > FROZEN_THRESHOLD).astype(int)
        true  = yr_pro[:, k]
        f1    = float(f1_score(true, preds, average="macro", zero_division=0))
        horizon_f1s[f"k{k+1}"] = round(f1, 4)

    # ── Print report ─────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("PHASE 7 — CONTROLLED FORECASTING PROGRESSION EXPERIMENT")
    print(SEP)
    print(f"\nScenario: BENIGN→ATTACK transition sequences (validation set)")
    print(f"Threshold: {FROZEN_THRESHOLD:.2f} (frozen from Phase 6 validation selection)")
    print(f"Window size: {window_sec}s | K={K} horizons ({K*window_sec}s ahead max)")
    print(f"\nProactive sequences (NOW=BENIGN, FUTURE=attack): {total:,}")

    print(f"\n--- Proactive Detection ---")
    print(f"  Sequences with alert BEFORE attack onset:  {proactive_count:,}  ({proactive_rate:.1%})")
    print(f"  Sequences with attack MISSED:              {(lead_times < 0).sum():,}")

    print(f"\n--- Lead Time Distribution (all sequences) ---")
    print(f"  Median lead time:       {median_lt:+.1f}s")
    print(f"  Mean lead time:         {mean_lt:+.1f}s")
    print(f"  P10 / P90:              {p10_lt:+.1f}s / {p90_lt:+.1f}s")
    print(f"  Median (positive only): {median_positive:+.1f}s  ({len(positive_lt):,} sequences)")

    print(f"\n--- Per-Horizon Alert Counts (among proactive sequences) ---")
    for k in range(K):
        print(f"  k={k+1} ({(k+1)*window_sec:3d}s ahead):  alert count = {per_horizon_detected[k]:,}")

    print(f"\n--- Per-Horizon Risk F1 on Proactive Sequences ---")
    for k, v in horizon_f1s.items():
        print(f"  {k}: Macro-F1 = {v:.4f}")

    # ── Criterion 3 (progression-scenario version) ───────────────────────────
    c3_median_passes = bool(median_lt >= 10.0)
    c3_rate_passes   = bool(proactive_rate >= 0.60)
    c3_passes        = c3_median_passes and c3_rate_passes

    print(f"\n--- Criterion 3 (Progression Scenario) ---")
    print(f"  Median lead time ≥ 10s:  {'✅ PASS' if c3_median_passes else '❌ FAIL'}  ({median_lt:.1f}s)")
    print(f"  Proactive rate ≥ 60%:    {'✅ PASS' if c3_rate_passes else '❌ FAIL'}  ({proactive_rate:.1%})")
    print(f"  Overall Criterion 3:     {'✅ PASS' if c3_passes else '❌ FAIL (valid negative result)'}")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "experiment": "Phase 7 — Controlled Forecasting Progression",
        "scenario": "validation BENIGN→ATTACK transition sequences",
        "threshold_used": FROZEN_THRESHOLD,
        "threshold_source": "frozen from Phase 6 validation selection",
        "window_seconds": window_sec,
        "K": K,
        "total_proactive_sequences": int(total),
        "proactive_detection_count": int(proactive_count),
        "proactive_detection_rate": round(proactive_rate, 4),
        "lead_time": {
            "median_all_sec": round(median_lt, 2),
            "mean_all_sec": round(mean_lt, 2),
            "p10_sec": round(p10_lt, 2),
            "p90_sec": round(p90_lt, 2),
            "median_positive_only_sec": round(median_positive, 2),
            "positive_count": int(len(positive_lt)),
        },
        "per_horizon_alert_counts": {f"k{k+1}": int(per_horizon_detected[k]) for k in range(K)},
        "per_horizon_risk_f1": horizon_f1s,
        "criterion_3_progression": {
            "median_passes": c3_median_passes,
            "rate_passes": c3_rate_passes,
            "overall_passes": c3_passes,
        },
    }

    out_path = reports_dir / "phase7_progression_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Phase 7 report saved to {out_path}")


if __name__ == "__main__":
    main()
