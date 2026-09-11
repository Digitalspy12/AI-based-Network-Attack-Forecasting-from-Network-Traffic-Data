#!/usr/bin/env python3
"""Phase 7.3 — Pre-Attack Feature Analysis.

For every sequence where current=BENIGN and future=ATTACK,
compare its 24-feature history against current=BENIGN and future=BENIGN.

Questions (from nextstep.md):
  1. Do any of the 24 features differ between pre-attack and pure-BENIGN histories?
  2. Which features are most discriminative (Mann-Whitney U, effect size)?
  3. Does the LSTM assign meaningfully different risk probabilities to pre-attack
     sequences vs pure-BENIGN sequences? (probability separation analysis)

Output:
  - Feature-wise statistics table
  - Top discriminative features ranked by effect size
  - LSTM probability distribution comparison
  - reports/phase7_3_feature_analysis.json
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.features.engineer import FEATURE_NAMES
from digitalspy.models.attention_lstm import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEP = "=" * 70


def run_inference_k1(model, X, device, batch_size=512):
    """Return (N,) k=1 risk probabilities."""
    model.eval()
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(device)
            rp, _, _ = model(xb)
            probs.append(rp[:, 0].cpu().numpy())
    return np.concatenate(probs)


def cohens_d(a, b):
    """Cohen's d effect size between two groups."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir    = config.resolve_path("models")
    reports_dir   = config.resolve_path("reports")
    lstm_cfg      = config.lstm()
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──────────────────────────────────────────────────────────
    model = build_model(lstm_cfg).to(device)
    model.load_state_dict(torch.load(models_dir / "lstm_checkpoint.pt", map_location=device))
    model.eval()

    # ── Load validation sequences ────────────────────────────────────────────
    val = np.load(processed_dir / "val_sequences.npz")
    X_val, y_risk_val = val["X"], val["y_risk"]

    # ── Partition into three groups ──────────────────────────────────────────
    current_benign  = y_risk_val[:, 0] == 0
    future_attack   = y_risk_val[:, 1:].sum(axis=1) > 0
    future_benign   = y_risk_val[:, 1:].sum(axis=1) == 0

    # Group A: BENIGN → ATTACK (proactive sequences)
    mask_A = current_benign & future_attack
    # Group B: BENIGN → BENIGN (pure-benign sequences)
    mask_B = current_benign & future_benign

    logger.info(f"Group A (pre-attack):   {mask_A.sum():,} sequences")
    logger.info(f"Group B (pure-BENIGN):  {mask_B.sum():,} sequences")

    X_A = X_val[mask_A]  # (N_A, 20, 24)
    X_B = X_val[mask_B]  # (N_B, 20, 24)

    # Use mean over the 20-step history as feature representation
    # Shape: (N, 24)
    feat_A = X_A.mean(axis=1)
    feat_B = X_B.mean(axis=1)

    # ── LSTM probability comparison ──────────────────────────────────────────
    logger.info("Running LSTM inference on both groups...")
    probs_A = run_inference_k1(model, X_A, device)
    probs_B = run_inference_k1(model, X_B, device)

    print(f"\n{SEP}")
    print("PHASE 7.3 — PRE-ATTACK FEATURE ANALYSIS")
    print(SEP)
    print(f"\nGroup A (BENIGN→ATTACK):  {len(X_A):,} sequences")
    print(f"Group B (BENIGN→BENIGN):  {len(X_B):,} sequences")

    # ── LSTM probability separation ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("LSTM k=1 RISK PROBABILITY SEPARATION")
    print(f"{'─'*70}")
    print(f"  Group A (pre-attack)  — mean={probs_A.mean():.4f}  median={np.median(probs_A):.4f}  P90={np.percentile(probs_A,90):.4f}")
    print(f"  Group B (pure-BENIGN) — mean={probs_B.mean():.4f}  median={np.median(probs_B):.4f}  P90={np.percentile(probs_B,90):.4f}")

    stat, pval = stats.mannwhitneyu(probs_A, probs_B, alternative="greater")
    d = cohens_d(probs_A, probs_B)
    print(f"\n  Mann-Whitney U (A>B): stat={stat:.1f}, p={pval:.4e}")
    print(f"  Cohen's d:            {d:.4f}  ({'large' if abs(d)>0.8 else 'medium' if abs(d)>0.5 else 'small' if abs(d)>0.2 else 'negligible'})")

    # Threshold-based pre-attack detection at k=1
    for t in [0.30, 0.50, 0.70, 0.90]:
        pdr_a = (probs_A > t).mean()
        fpr_b = (probs_B > t).mean()
        print(f"  At threshold {t:.2f}: pre-attack alert rate={pdr_a:.1%}, false-BENIGN rate={fpr_b:.1%}")

    # ── Feature-wise statistical comparison ─────────────────────────────────
    print(f"\n{'─'*70}")
    print("FEATURE-WISE ANALYSIS (mean over 20-step history)")
    print(f"{'─'*70}")
    print(f"{'Feature':<25} {'Mean_A':>9} {'Mean_B':>9} {'Cohen_d':>9} {'p_value':>12}  {'Sig?':>5}")
    print("-" * 74)

    feature_results = []
    for j, fname in enumerate(FEATURE_NAMES):
        a_vals = feat_A[:, j]
        b_vals = feat_B[:, j]
        d_val  = cohens_d(a_vals, b_vals)
        _, pv  = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
        sig    = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
        print(f"  {fname:<23} {a_vals.mean():>9.4f} {b_vals.mean():>9.4f} {d_val:>9.4f} {pv:>12.4e}  {sig:>5}")
        feature_results.append({
            "feature": fname,
            "mean_preattack": round(float(a_vals.mean()), 6),
            "mean_benign": round(float(b_vals.mean()), 6),
            "cohens_d": round(d_val, 4),
            "p_value": float(pv),
            "significant_001": bool(pv < 0.001),
        })

    # ── Top discriminative features ──────────────────────────────────────────
    ranked = sorted(feature_results, key=lambda x: abs(x["cohens_d"]), reverse=True)
    print(f"\n{'─'*70}")
    print("TOP 10 DISCRIMINATIVE FEATURES (by |Cohen's d|)")
    print(f"{'─'*70}")
    for i, r in enumerate(ranked[:10], 1):
        direction = "↑ pre-attack" if r["mean_preattack"] > r["mean_benign"] else "↓ pre-attack"
        print(f"  {i:2d}. {r['feature']:<25}  d={r['cohens_d']:>7.4f}  {direction}")

    # ── Summary verdict ──────────────────────────────────────────────────────
    sig_features = [r for r in feature_results if r["significant_001"] and abs(r["cohens_d"]) > 0.2]
    large_effect  = [r for r in feature_results if abs(r["cohens_d"]) > 0.5]

    print(f"\n{'─'*70}")
    print("VERDICT")
    print(f"{'─'*70}")
    print(f"  Features with p<0.001 AND |d|>0.2:  {len(sig_features):,}")
    print(f"  Features with |d|>0.5 (medium+):    {len(large_effect):,}")
    if len(large_effect) >= 3:
        print("  ✅ Discriminative signal EXISTS — pre-attack features are detectable")
        print("     → Feature engineering (delta/trend features) is scientifically justified")
    elif len(sig_features) >= 3:
        print("  ⚠️  Weak signal — some statistical difference but small effect size")
        print("     → Delta features may help amplify the signal")
    else:
        print("  ❌ No meaningful signal — pre-attack and BENIGN histories look identical")
        print("     → Model architecture change may be more important than feature engineering")

    # ── Save ─────────────────────────────────────────────────────────────────
    report = {
        "experiment": "Phase 7.3 — Pre-Attack Feature Analysis",
        "group_A_size": int(len(X_A)),
        "group_B_size": int(len(X_B)),
        "lstm_probability_separation": {
            "group_A_mean": round(float(probs_A.mean()), 4),
            "group_A_median": round(float(np.median(probs_A)), 4),
            "group_B_mean": round(float(probs_B.mean()), 4),
            "group_B_median": round(float(np.median(probs_B)), 4),
            "mann_whitney_p": float(pval),
            "cohens_d": round(d, 4),
        },
        "feature_analysis": feature_results,
        "top_discriminative_features": ranked[:10],
        "summary": {
            "n_significant_medium_effect": len(sig_features),
            "n_large_effect": len(large_effect),
        },
    }
    out = reports_dir / "phase7_3_feature_analysis.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Phase 7.3 report saved to {out}")


if __name__ == "__main__":
    main()
