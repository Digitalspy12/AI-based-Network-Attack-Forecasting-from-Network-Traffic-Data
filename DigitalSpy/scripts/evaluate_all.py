#!/usr/bin/env python3
"""Phase 10 — Full Evaluation: Experiments A, B, C.

Evaluates all three experiments per IMPLEMENTATION.md §9 and §17:

Experiment A: S_t → LR → Y_t  (current detection)
Experiment B: S_t → LR → Y_{t+1}  vs  H_t → LSTM → Y_{t+1}
Experiment C: Markov P(Z_{t+k}|Z_t)  vs  LSTM P(Y_{t+k}|H_t) for k=1..5

Success Criteria (pre-fixed, IMPLEMENTATION.md §17):
  Criterion 1: LSTM k=1 risk macro-F1 ≥ LR k=1 + 5pp, FPR increase ≤ 2pp
  Criterion 2: LSTM F1_K ≥ Markov F1_K + 5pp (tactic)
  Criterion 3: median lead time ≥ 10s, ≥60% positive (see evaluate_lead_time.py)

Brier score and ECE also computed (§19).

CRITICAL: Test data is NOT used for threshold selection or tuning.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.baselines.logistic import LogisticBaseline, evaluate_multiclass
from digitalspy.features.engineer import FEATURE_NAMES
from digitalspy.markov.model import MarkovWorldModel
from digitalspy.models.attention_lstm import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error for binary predictions."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_conf - avg_acc)
    return float(ece)


def lstm_predict_all(model, X, K, num_classes, device, batch_size=256):
    """Run LSTM inference on all sequences."""
    model.eval()
    risk_all, tactic_all = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(device)
            risk_probs, tactic_logits, _ = model(xb)
            risk_all.append(risk_probs.cpu().numpy())
            tactic_softmax = torch.softmax(tactic_logits, dim=-1).cpu().numpy()
            tactic_all.append(tactic_softmax)
    return np.vstack(risk_all), np.concatenate(tactic_all, axis=0)


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir = config.resolve_path("models")
    reports_dir = config.resolve_path("reports")
    lstm_cfg = config.lstm()
    K = lstm_cfg["forecast"]["K"]
    num_classes = lstm_cfg["forecast"]["heads"]["tactic"]["num_classes"]
    state_names = config.labels()["states"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load windows ────────────────────────────────────────────────────────
    state_path = processed_dir / "state_windows.parquet"
    all_w = pd.read_parquet(state_path)
    train_w = all_w[all_w["split"] == "train"].reset_index(drop=True)
    val_w = all_w[all_w["split"] == "validation"].reset_index(drop=True)
    test_w = all_w[all_w["split"] == "test"].reset_index(drop=True)

    X_te = test_w[FEATURE_NAMES].values
    y_risk_te = (test_w["z_t"] != "BENIGN").astype(int).values
    y_tactic_te = test_w["z_t_idx"].values

    # ── Load LSTM sequences ──────────────────────────────────────────────────
    seq_path = processed_dir / "test_sequences.npz"
    if not seq_path.exists():
        logger.error("test_sequences.npz not found. Run build_states.py and train_lstm.py first.")
        sys.exit(1)
    seq_data = np.load(seq_path)
    X_seq_te, y_risk_seq_te, y_tactic_seq_te = seq_data["X"], seq_data["y_risk"], seq_data["y_tactic"]

    # ── Load models ──────────────────────────────────────────────────────────
    lr_curr_risk = LogisticBaseline.load(models_dir / "logistic_current_risk.pkl", task="binary")
    lr_os_risk = LogisticBaseline.load(models_dir / "logistic_onestep_risk.pkl", task="binary")
    lr_os_tactic = LogisticBaseline.load(models_dir / "logistic_onestep_tactic.pkl", task="multiclass", class_names=state_names)
    markov = MarkovWorldModel.load(models_dir / "markov_matrix.npy")

    lstm_model = build_model(lstm_cfg).to(device)
    checkpoint_path = models_dir / "lstm_checkpoint.pt"
    if not checkpoint_path.exists():
        logger.error("lstm_checkpoint.pt not found. Run train_lstm.py first.")
        sys.exit(1)
    lstm_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    all_metrics = {}

    # ── Experiment A: Current Detection ─────────────────────────────────────
    logger.info("=== Experiment A: Current Detection ===")
    m_a = lr_curr_risk.evaluate(X_te, y_risk_te, label="A_current_detection")
    all_metrics["experiment_A"] = m_a
    logger.info(f"LR current: macro-F1={m_a['macro_f1']:.4f}, FPR={m_a['fpr']:.4f}")

    # ── Experiment B: One-Step Forecasting ──────────────────────────────────
    logger.info("=== Experiment B: One-Step Forecasting ===")

    def _build_onestep_targets(windows: pd.DataFrame):
        """Build shifted targets: X[i] → y[i+1].

        Groups by source_ip+window_start when available; falls back to
        global consecutive-row pairs sorted by window_idx (CIC-IDS-2017
        has no IP columns).
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
            return np.vstack(X_list), np.concatenate(y_risk_list), np.concatenate(y_tactic_list)
        else:
            sort_col = "window_idx" if "window_idx" in windows.columns else None
            w = windows.sort_values(sort_col).reset_index(drop=True) if sort_col else windows.reset_index(drop=True)
            if len(w) < 2:
                return np.zeros((0, len(FEATURE_NAMES)), dtype=float), np.zeros(0, dtype=int), np.zeros(0, dtype=int)
            return (
                w[FEATURE_NAMES].values[:-1],
                (w["z_t"].values[1:] != "BENIGN").astype(int),
                w["z_t_idx"].values[1:],
            )

    # B-LR: build one-step test targets
    X_te1, y_risk_te1, _ = _build_onestep_targets(test_w)
    m_b_lr = lr_os_risk.evaluate(X_te1, y_risk_te1, label="B_LR_onestep_risk")
    all_metrics["experiment_B_LR"] = m_b_lr
    logger.info(f"LR one-step risk: macro-F1={m_b_lr['macro_f1']:.4f}, FPR={m_b_lr['fpr']:.4f}")

    # B-LSTM: k=1 predictions
    lstm_risk_probs, lstm_tactic_probs = lstm_predict_all(lstm_model, X_seq_te, K, num_classes, device)
    lstm_k1_risk_preds = (lstm_risk_probs[:, 0] > 0.5).astype(int)
    lstm_k1_risk_f1 = float(f1_score(y_risk_seq_te[:, 0], lstm_k1_risk_preds, average="macro", zero_division=0))
    all_metrics["experiment_B_LSTM_k1_macro_f1"] = round(lstm_k1_risk_f1, 4)
    logger.info(f"LSTM one-step risk: macro-F1={lstm_k1_risk_f1:.4f}")

    # Criterion 1 check
    diff_pp = (lstm_k1_risk_f1 - m_b_lr["macro_f1"]) * 100
    all_metrics["criterion_1"] = {
        "lstm_k1_f1": round(lstm_k1_risk_f1, 4),
        "lr_k1_f1": round(m_b_lr["macro_f1"], 4),
        "difference_pp": round(diff_pp, 2),
        "passes": bool(diff_pp >= 5.0),
        "threshold": 5.0,
    }
    logger.info(f"Criterion 1: diff={diff_pp:+.2f}pp → {'PASS ✓' if diff_pp >= 5.0 else 'FAIL ✗'}")

    # ── Experiment C: Multi-Step Tactic Forecasting ──────────────────────────
    logger.info("=== Experiment C: Multi-Step Tactic Forecasting ===")

    # Markov k-step predictions
    markov_f1s = []
    lstm_tactic_f1s = []
    for k in range(1, K + 1):
        # Markov
        z0 = y_tactic_seq_te[:, 0] if y_tactic_seq_te.shape[1] > 0 else np.zeros(len(y_tactic_seq_te), dtype=int)
        markov_preds = np.array([int(np.argmax(markov.predict_k_step(int(z), k)[-1])) for z in z0])
        y_true_k = y_tactic_seq_te[:, k - 1]
        m_f1 = float(f1_score(y_true_k, markov_preds, average="macro", zero_division=0))
        markov_f1s.append(m_f1)

        # LSTM
        lstm_tactic_k = np.argmax(lstm_tactic_probs[:, k - 1, :], axis=-1)
        l_f1 = float(f1_score(y_true_k, lstm_tactic_k, average="macro", zero_division=0))
        lstm_tactic_f1s.append(l_f1)
        logger.info(f"  k={k}: Markov={m_f1:.4f}, LSTM={l_f1:.4f}")

    F1_K_markov = float(np.mean(markov_f1s))
    F1_K_lstm = float(np.mean(lstm_tactic_f1s))
    diff_c2 = (F1_K_lstm - F1_K_markov) * 100

    all_metrics["experiment_C"] = {
        "markov_f1_per_k": {f"k{k+1}": round(v, 4) for k, v in enumerate(markov_f1s)},
        "lstm_f1_per_k": {f"k{k+1}": round(v, 4) for k, v in enumerate(lstm_tactic_f1s)},
        "F1_K_markov": round(F1_K_markov, 4),
        "F1_K_lstm": round(F1_K_lstm, 4),
    }
    all_metrics["criterion_2"] = {
        "lstm_F1_K": round(F1_K_lstm, 4),
        "markov_F1_K": round(F1_K_markov, 4),
        "difference_pp": round(diff_c2, 2),
        "passes": bool(diff_c2 >= 5.0),
        "threshold": 5.0,
    }
    logger.info(f"Criterion 2: F1_K diff={diff_c2:+.2f}pp → {'PASS ✓' if diff_c2 >= 5.0 else 'FAIL ✗'}")

    # ── Calibration (§19) ────────────────────────────────────────────────────
    brier = float(brier_score_loss(y_risk_seq_te[:, 0], lstm_risk_probs[:, 0]))
    ece = expected_calibration_error(y_risk_seq_te[:, 0], lstm_risk_probs[:, 0])
    all_metrics["calibration"] = {
        "brier_score_k1": round(brier, 4),
        "ece_k1": round(ece, 4),
        "note": "Brier score closer to 0 is better. ECE closer to 0 means better-calibrated probabilities.",
    }
    logger.info(f"Calibration: Brier={brier:.4f}, ECE={ece:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    report_path = reports_dir / "final_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL EVALUATION REPORT")
    print("=" * 70)
    print(f"\nCriterion 1 (one-step risk): {'PASS ✓' if all_metrics['criterion_1']['passes'] else 'FAIL ✗'}")
    print(f"  LSTM k=1 F1: {lstm_k1_risk_f1:.4f}  LR k=1 F1: {m_b_lr['macro_f1']:.4f}  Δ={diff_pp:+.2f}pp")
    print(f"\nCriterion 2 (multi-step tactic): {'PASS ✓' if all_metrics['criterion_2']['passes'] else 'FAIL ✗'}")
    print(f"  LSTM F1_K: {F1_K_lstm:.4f}  Markov F1_K: {F1_K_markov:.4f}  Δ={diff_c2:+.2f}pp")
    print(f"\nCriterion 3 (lead time): See reports/lead_time_metrics.json")
    print(f"\n✓ Report saved to {report_path}")


if __name__ == "__main__":
    main()
