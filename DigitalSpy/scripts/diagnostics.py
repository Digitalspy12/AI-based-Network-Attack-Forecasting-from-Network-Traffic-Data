#!/usr/bin/env python3
"""Diagnostic inspection — nextstep.md checklist.

1. State distributions per split (train / val / test)
2. Raw Markov transition counts (before Laplace smoothing)
3. LSTM k=1 confusion matrix
4. Per-class F1 for LSTM k=1
5. LSTM prediction distribution (class collapse check)
"""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.models.attention_lstm import build_model
from digitalspy.states.labels import index_to_label

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEP = "=" * 60


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir    = config.resolve_path("models")
    reports_dir   = config.resolve_path("reports")

    label_cfg   = config.labels()
    state_names = label_cfg["states"]   # ['BENIGN','RECON','INITIAL_ACCESS',...]
    num_classes = len(state_names)

    # ── Load state windows ────────────────────────────────────────────────────
    parquet_path = processed_dir / "state_windows.parquet"
    all_windows  = pd.read_parquet(parquet_path)

    train_w = all_windows[all_windows["split"] == "train"]
    val_w   = all_windows[all_windows["split"] == "validation"]
    test_w  = all_windows[all_windows["split"] == "test"]

    # ── 1. State distributions ────────────────────────────────────────────────
    section("1. STATE DISTRIBUTIONS PER SPLIT")
    for split_name, df in [("TRAIN", train_w), ("VALIDATION", val_w), ("TEST", test_w)]:
        counts = df["z_t"].value_counts()
        total  = len(df)
        print(f"\n{split_name} ({total:,} windows):")
        for s in state_names:
            n   = counts.get(s, 0)
            pct = 100 * n / total if total > 0 else 0
            print(f"  {s:<20} {n:>8,}  ({pct:5.2f}%)")

    # ── 2. Raw Markov transition counts ───────────────────────────────────────
    section("2. RAW MARKOV TRANSITION COUNTS (before Laplace smoothing)")

    sort_col = "window_idx" if "window_idx" in train_w.columns else None
    tw = train_w.sort_values(sort_col).reset_index(drop=True) if sort_col else train_w.reset_index(drop=True)
    z_seq = tw["z_t_idx"].values

    raw_counts = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(z_seq) - 1):
        zi, zj = int(z_seq[i]), int(z_seq[i + 1])
        if 0 <= zi < num_classes and 0 <= zj < num_classes:
            raw_counts[zi, zj] += 1

    raw_df = pd.DataFrame(raw_counts, index=state_names, columns=state_names)
    row_totals = raw_counts.sum(axis=1)
    print("\nRaw counts matrix (rows=from, cols=to):")
    print(raw_df.to_string())
    print("\nTotal observed outgoing transitions per state:")
    for i, s in enumerate(state_names):
        print(f"  {s:<20} {row_totals[i]:>8,} transitions")

    # ── 3–5: Load LSTM + sequences ────────────────────────────────────────────
    lstm_cfg   = config.lstm()
    K          = lstm_cfg["forecast"]["K"]
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path  = models_dir / "lstm_checkpoint.pt"

    if not ckpt_path.exists():
        print("\n[WARNING] lstm_checkpoint.pt not found — skipping LSTM diagnostics.")
        return

    model = build_model(lstm_cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_npz = np.load(processed_dir / "test_sequences.npz")
    X_te, yr_te, yt_te = test_npz["X"], test_npz["y_risk"], test_npz["y_tactic"]

    # Collect k=1 predictions
    batch_size = 512
    all_tactic_pred_k1, all_tactic_true_k1 = [], []
    all_risk_pred_k1,   all_risk_true_k1   = [], []

    with torch.no_grad():
        for i in range(0, len(X_te), batch_size):
            xb = torch.FloatTensor(X_te[i:i+batch_size]).to(device)
            risk_probs, tactic_logits, _ = model(xb)

            # k=1 is index 0
            risk_pred   = (risk_probs[:, 0].cpu().numpy() > 0.5).astype(int)
            tactic_pred = torch.argmax(tactic_logits[:, 0, :], dim=-1).cpu().numpy()

            all_risk_pred_k1.append(risk_pred)
            all_risk_true_k1.append(yr_te[i:i+batch_size, 0])
            all_tactic_pred_k1.append(tactic_pred)
            all_tactic_true_k1.append(yt_te[i:i+batch_size, 0])

    tp = np.concatenate(all_tactic_pred_k1)
    tt = np.concatenate(all_tactic_true_k1)

    # ── 3. LSTM k=1 Confusion Matrix ─────────────────────────────────────────
    section("3. LSTM k=1 TACTIC CONFUSION MATRIX")
    cm = confusion_matrix(tt, tp, labels=list(range(num_classes)))
    cm_df = pd.DataFrame(cm, index=[f"True:{s}" for s in state_names],
                             columns=[f"Pred:{s[:6]}" for s in state_names])
    print(cm_df.to_string())

    # ── 4. Per-class F1 ───────────────────────────────────────────────────────
    section("4. PER-CLASS F1 — LSTM k=1")
    report = classification_report(
        tt, tp,
        labels=list(range(num_classes)),
        target_names=state_names,
        zero_division=0,
    )
    print(report)

    # ── 5. Prediction distribution ────────────────────────────────────────────
    section("5. LSTM k=1 PREDICTION DISTRIBUTION (class collapse check)")
    pred_counts = np.bincount(tp, minlength=num_classes)
    total_preds = len(tp)
    print(f"\nTotal test predictions: {total_preds:,}")
    for i, s in enumerate(state_names):
        pct = 100 * pred_counts[i] / total_preds
        bar = "█" * int(pct / 2)
        print(f"  {s:<20} {pred_counts[i]:>8,}  ({pct:5.2f}%)  {bar}")

    # True distribution for comparison
    true_counts = np.bincount(tt, minlength=num_classes)
    print(f"\nTrue test distribution:")
    for i, s in enumerate(state_names):
        pct = 100 * true_counts[i] / total_preds
        bar = "█" * int(pct / 2)
        print(f"  {s:<20} {true_counts[i]:>8,}  ({pct:5.2f}%)  {bar}")

    # Save summary JSON
    diag = {
        "state_distributions": {
            split: {s: int(df["z_t"].value_counts().get(s, 0)) for s in state_names}
            for split, df in [("train", train_w), ("validation", val_w), ("test", test_w)]
        },
        "markov_raw_counts": {
            state_names[i]: {state_names[j]: int(raw_counts[i, j]) for j in range(num_classes)}
            for i in range(num_classes)
        },
        "markov_row_totals": {state_names[i]: int(row_totals[i]) for i in range(num_classes)},
        "lstm_k1_pred_distribution": {state_names[i]: int(pred_counts[i]) for i in range(num_classes)},
        "lstm_k1_true_distribution": {state_names[i]: int(true_counts[i]) for i in range(num_classes)},
        "lstm_k1_macro_f1": round(float(f1_score(tt, tp, average="macro", zero_division=0)), 4),
    }
    out_path = reports_dir / "diagnostics.json"
    with open(out_path, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"\n✓ Diagnostics saved to {out_path}")


if __name__ == "__main__":
    main()
