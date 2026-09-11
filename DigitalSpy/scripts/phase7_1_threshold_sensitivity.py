#!/usr/bin/env python3
"""Phase 7.1 — Threshold Sensitivity with Independent Val-A / Val-B Split.

Fixes from nextstep.md:
  1. Val-A (first 65%) → threshold selection only
  2. Val-B (last 35%) → proactive forecasting evaluation
  3. Clean lead-time reporting: proactive / reactive / missed — NO mixed negatives
  4. Sweep thresholds 0.30 → 0.90

Metrics per threshold:
  - Val-A: Precision, Recall, Macro-F1, FPR (for selection)
  - Val-B: PDR, median proactive lead time, median reactive delay, missed rate
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.models.attention_lstm import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEP = "=" * 70
THRESHOLDS = np.round(np.arange(0.30, 0.95, 0.05), 2)
VAL_A_FRAC  = 0.65   # fraction of val for threshold calibration


def run_inference(model, X, device, batch_size=512):
    """Return (N, K) risk probability array."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(device)
            rp, _, _ = model(xb)
            chunks.append(rp.cpu().numpy())
    return np.vstack(chunks)


def compute_detection_metrics(risk_probs, y_risk, threshold, window_sec):
    """
    Compute clean 3-way lead-time breakdown:
        PROACTIVE  — alert fired BEFORE attack onset (lead > 0)
        REACTIVE   — alert fired AT or AFTER onset (lead <= 0 but detected)
        MISSED     — no alert fired at any horizon
    """
    K = risk_probs.shape[1]
    proactive_lts = []
    reactive_delays = []
    missed = 0
    no_attack = 0

    for i in range(len(y_risk)):
        onset_ks = np.where(y_risk[i] == 1)[0]
        if len(onset_ks) == 0:
            no_attack += 1
            continue

        attack_k = int(onset_ks[0])
        attack_t = (attack_k + 1) * window_sec

        alert_ks = np.where(risk_probs[i] > threshold)[0]
        if len(alert_ks) == 0:
            missed += 1
            continue

        alert_k = int(alert_ks[0])
        alert_t = (alert_k + 1) * window_sec
        lead    = attack_t - alert_t   # +ve = proactive, -ve = reactive

        if lead > 0:
            proactive_lts.append(lead)
        else:
            reactive_delays.append(-lead)   # store as positive delay

    total_eligible = len(y_risk) - no_attack
    pdr = len(proactive_lts) / total_eligible if total_eligible > 0 else 0.0

    return {
        "total_eligible": int(total_eligible),
        "proactive": {
            "count": len(proactive_lts),
            "rate": round(pdr, 4),
            "median_lead_sec": round(float(np.median(proactive_lts)), 2) if proactive_lts else 0.0,
            "mean_lead_sec":   round(float(np.mean(proactive_lts)), 2) if proactive_lts else 0.0,
            "p90_lead_sec":    round(float(np.percentile(proactive_lts, 90)), 2) if proactive_lts else 0.0,
        },
        "reactive": {
            "count": len(reactive_delays),
            "rate": round(len(reactive_delays) / total_eligible, 4) if total_eligible > 0 else 0.0,
            "median_delay_sec": round(float(np.median(reactive_delays)), 2) if reactive_delays else 0.0,
        },
        "missed": {
            "count": int(missed),
            "rate": round(missed / total_eligible, 4) if total_eligible > 0 else 0.0,
        },
        "no_attack_sequences": int(no_attack),
    }


def val_f1_at_threshold(risk_probs_k1, y_risk_k1, threshold):
    preds = (risk_probs_k1 > threshold).astype(int)
    return float(f1_score(y_risk_k1, preds, average="macro", zero_division=0))


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir    = config.resolve_path("models")
    reports_dir   = config.resolve_path("reports")
    sys_cfg       = config.system()
    lstm_cfg      = config.lstm()
    window_sec    = sys_cfg["windowing"]["window_seconds"]
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────
    model = build_model(lstm_cfg).to(device)
    model.load_state_dict(torch.load(models_dir / "lstm_checkpoint.pt", map_location=device))
    model.eval()
    logger.info(f"Model loaded. Device: {device}")

    # ── Load validation sequences ───────────────────────────────────────────
    val = np.load(processed_dir / "val_sequences.npz")
    X_val, y_risk_val = val["X"], val["y_risk"]
    N = len(X_val)

    split_n = int(N * VAL_A_FRAC)
    X_a,    yr_a    = X_val[:split_n],    y_risk_val[:split_n]
    X_b,    yr_b    = X_val[split_n:],    y_risk_val[split_n:]

    logger.info(f"Val-A (threshold calibration): {len(X_a):,} sequences")
    logger.info(f"Val-B (proactive evaluation):  {len(X_b):,} sequences")

    # ── Run inference on both halves ─────────────────────────────────────────
    rp_a = run_inference(model, X_a, device)
    rp_b = run_inference(model, X_b, device)

    # ── Identify proactive sequences in Val-B ────────────────────────────────
    current_benign_b = yr_b[:, 0] == 0
    future_attack_b  = yr_b[:, 1:].sum(axis=1) > 0
    proactive_mask_b = current_benign_b & future_attack_b
    proactive_idx_b  = np.where(proactive_mask_b)[0]

    logger.info(f"Val-B proactive sequences (NOW=BENIGN, FUTURE=attack): {proactive_mask_b.sum():,}")

    rp_b_pro  = rp_b[proactive_idx_b]
    yr_b_pro  = yr_b[proactive_idx_b]

    # ── Threshold sweep ───────────────────────────────────────────────────────
    results = []
    best_thresh = 0.5
    best_f1     = 0.0

    print(f"\n{SEP}")
    print("PHASE 7.1 — THRESHOLD SENSITIVITY")
    print(SEP)
    print(f"\nVal-A: {len(X_a):,} | Val-B: {len(X_b):,} | Val-B proactive: {proactive_mask_b.sum():,}")
    print(f"\n{'Threshold':>10} | {'Val-A F1':>9} | {'PDR':>7} | {'Median+LT':>10} | {'Reactive%':>10} | {'Missed%':>8}")
    print("-" * 70)

    for t in THRESHOLDS:
        # Val-A F1 (for threshold selection)
        valf1 = val_f1_at_threshold(rp_a[:, 0], yr_a[:, 0], t)
        if valf1 > best_f1:
            best_f1     = valf1
            best_thresh = float(t)

        # Val-B proactive metrics
        det = compute_detection_metrics(rp_b_pro, yr_b_pro, float(t), window_sec)
        pdr = det["proactive"]["rate"]
        med_lt  = det["proactive"]["median_lead_sec"]
        react_r = det["reactive"]["rate"]
        miss_r  = det["missed"]["rate"]

        print(f"{t:>10.2f} | {valf1:>9.4f} | {pdr:>7.3f} | {med_lt:>+10.1f}s | {react_r:>9.1%} | {miss_r:>7.1%}")

        results.append({
            "threshold": float(t),
            "val_a_macro_f1": round(valf1, 4),
            "val_b_proactive": det,
        })

    # ── Best threshold results ────────────────────────────────────────────────
    print(f"\n→ Best threshold by Val-A F1: {best_thresh:.2f}  (F1={best_f1:.4f})")

    best_det = compute_detection_metrics(rp_b_pro, yr_b_pro, best_thresh, window_sec)
    print(f"\n{SEP}")
    print(f"BEST THRESHOLD = {best_thresh:.2f} — PROACTIVE METRICS (Val-B)")
    print(SEP)
    p = best_det["proactive"]
    r = best_det["reactive"]
    m = best_det["missed"]
    print(f"\n  Eligible attack sequences:  {best_det['total_eligible']:,}")
    print(f"\n  PROACTIVE (alert before onset):")
    print(f"    Count:          {p['count']:,}  ({p['rate']:.1%})")
    print(f"    Median lead:    {p['median_lead_sec']:+.1f}s")
    print(f"    Mean lead:      {p['mean_lead_sec']:+.1f}s")
    print(f"    P90 lead:       {p['p90_lead_sec']:+.1f}s")
    print(f"\n  REACTIVE (alert at or after onset):")
    print(f"    Count:          {r['count']:,}  ({r['rate']:.1%})")
    print(f"    Median delay:   {r['median_delay_sec']:+.1f}s")
    print(f"\n  MISSED (no alert at any horizon):")
    print(f"    Count:          {m['count']:,}  ({m['rate']:.1%})")

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "experiment": "Phase 7.1 — Threshold Sensitivity",
        "val_a_size": int(len(X_a)),
        "val_b_size": int(len(X_b)),
        "val_b_proactive_sequences": int(proactive_mask_b.sum()),
        "window_seconds": window_sec,
        "val_a_fraction": VAL_A_FRAC,
        "best_threshold_by_val_a_f1": best_thresh,
        "best_val_a_f1": round(best_f1, 4),
        "threshold_sweep": results,
        "best_threshold_val_b_metrics": best_det,
    }
    out = reports_dir / "phase7_1_threshold_sensitivity.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Phase 7.1 report saved to {out}")
    return best_thresh


if __name__ == "__main__":
    main()
