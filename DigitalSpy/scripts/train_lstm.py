#!/usr/bin/env python3
"""Phase 5 — Train Attention-LSTM.

Trains the frozen Attention-LSTM architecture on 20×24 sequences.
Outputs:
    models/lstm_checkpoint.pt
    reports/lstm_val_curves.json
    reports/lstm_metrics.json
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
from digitalspy.models.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_sequences(tag: str, processed_dir: Path) -> tuple:
    path = processed_dir / f"{tag}_sequences.npz"
    if not path.exists():
        logger.error(f"{path} not found. Run build_states.py first.")
        sys.exit(1)
    data = np.load(path)
    return data["X"], data["y_risk"], data["y_tactic"]


def evaluate_lstm(model, X, y_risk, y_tactic, K, num_classes, device, label):
    """Evaluate LSTM on a dataset split."""
    model.eval()
    batch_size = 256
    all_risk_preds, all_tactic_preds = [], []
    all_risk_true, all_tactic_true = [], []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(device)
            risk_probs, tactic_logits, _ = model(xb)

            risk_pred = (risk_probs.cpu().numpy() > 0.5).astype(int)  # (batch, K)
            tactic_pred = torch.argmax(tactic_logits, dim=-1).cpu().numpy()  # (batch, K)

            all_risk_preds.append(risk_pred)
            all_tactic_preds.append(tactic_pred)
            all_risk_true.append(y_risk[i:i+batch_size])
            all_tactic_true.append(y_tactic[i:i+batch_size])

    risk_pred_all = np.vstack(all_risk_preds)
    tactic_pred_all = np.vstack(all_tactic_preds)
    risk_true_all = np.vstack(all_risk_true)
    tactic_true_all = np.vstack(all_tactic_true)

    metrics = {"label": label, "k_step_risk_f1": {}, "k_step_tactic_f1": {}}
    risk_f1s, tactic_f1s = [], []

    for k in range(K):
        rf1 = float(f1_score(risk_true_all[:, k], risk_pred_all[:, k], average="macro", zero_division=0))
        tf1 = float(f1_score(tactic_true_all[:, k], tactic_pred_all[:, k], average="macro", zero_division=0))
        metrics["k_step_risk_f1"][f"k{k+1}"] = round(rf1, 4)
        metrics["k_step_tactic_f1"][f"k{k+1}"] = round(tf1, 4)
        risk_f1s.append(rf1)
        tactic_f1s.append(tf1)

    metrics["F1_K_risk"] = round(float(np.mean(risk_f1s)), 4)
    metrics["F1_K_tactic"] = round(float(np.mean(tactic_f1s)), 4)
    metrics["k1_risk_f1"] = metrics["k_step_risk_f1"]["k1"]
    return metrics


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir = config.resolve_path("models")
    reports_dir = config.resolve_path("reports")
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    lstm_cfg = config.lstm()
    K = lstm_cfg["forecast"]["K"]
    num_classes = lstm_cfg["forecast"]["heads"]["tactic"]["num_classes"]

    # Load sequences
    X_tr, yr_tr, yt_tr = load_sequences("train", processed_dir)
    X_va, yr_va, yt_va = load_sequences("val", processed_dir)
    X_te, yr_te, yt_te = load_sequences("test", processed_dir)

    logger.info(f"Train: {len(X_tr):,} seqs | Val: {len(X_va):,} | Test: {len(X_te):,}")

    if len(X_tr) == 0:
        logger.error("No training sequences. Check build_states.py output.")
        sys.exit(1)

    # Train
    checkpoint_path = models_dir / "lstm_checkpoint.pt"
    val_curves_path = reports_dir / "lstm_val_curves.json"

    model, curves = train(
        X_tr, yr_tr, yt_tr,
        X_va, yr_va, yt_va,
        checkpoint_path=checkpoint_path,
        val_curves_path=val_curves_path,
        cfg=lstm_cfg,
    )

    # Evaluate on test set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test_metrics = evaluate_lstm(model, X_te, yr_te, yt_te, K, num_classes, device, "test")
    val_metrics = evaluate_lstm(model, X_va, yr_va, yt_va, K, num_classes, device, "validation")

    all_metrics = {
        "validation": val_metrics,
        "test": test_metrics,
        "best_epoch": curves["best_epoch"],
        "best_val_loss": round(curves["best_val_loss"], 6),
    }

    report_path = reports_dir / "lstm_metrics.json"
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("ATTENTION-LSTM EVALUATION")
    print("=" * 60)
    print(f"\nTest k=1 risk macro-F1: {test_metrics['k1_risk_f1']:.4f}")
    print(f"Test F1_K tactic:       {test_metrics['F1_K_tactic']:.4f}")
    print(f"\nPer-horizon risk F1: {test_metrics['k_step_risk_f1']}")
    print(f"Per-horizon tactic F1: {test_metrics['k_step_tactic_f1']}")
    print(f"\n✓ Phase 5 complete. Checkpoint: {checkpoint_path}")
    print(f"  Compare to Logistic k=1 (reports/baseline_metrics.json → B1)")


if __name__ == "__main__":
    main()
