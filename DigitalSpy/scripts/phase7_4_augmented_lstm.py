#!/usr/bin/env python3
"""Phase 7.4 — Temporal Delta/Trend Feature Augmentation + LSTM Retrain.

nextstep.md: "The model currently sees levels; forecasting may need trajectories."

Evidence from Phase 7.3:
  - std_iat (d=0.75), max_iat (d=0.60), mean_iat (d=0.42) are strongest precursors
  - 11 features statistically significant (p<0.001, |d|>0.2)
  - Signal exists but is weak in levels → amplify with deltas and trends

What we add (per nextstep.md):
  For each timestep t in the 20-step sequence, for all 24 features:
    1. Levels (original): x_t
    2. Rate change (Deltas): Δx_t = x_t - x_{t-1}
    3. Acceleration (Delta of Deltas): Δ²x_t = Δx_t - Δx_{t-1}
    4. Temporal volatility (Rolling Std): std(x_{t-4} ... x_t)
    5. Rolling Max: max(x_{t-4} ... x_t)

  New input size: 24 * 5 = 120 features per timestep.

Outputs:
  models/lstm_augmented_checkpoint.pt
  reports/phase7_4_augmented_metrics.json
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from digitalspy import config
from digitalspy.models.attention_lstm import AttentionLSTM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEP = "=" * 70

# ── Augmented architecture constants ─────────────────────────────────────────
AUG_INPUT_SIZE = 120   # 24 original + 4 derived sets (24*5)
HIDDEN_SIZE    = 64
NUM_LAYERS     = 1
DROPOUT        = 0.20
K              = 5
NUM_CLASSES    = 6

# Val split for threshold calibration vs proactive evaluation
VAL_A_FRAC     = 0.65
THRESHOLD      = 0.30   # best threshold from Phase 7.1


def add_trend_features(X: np.ndarray) -> np.ndarray:
    """Augment (N, T, 24) sequences with rich temporal features.

    Returns: (N, T, 120)
    Includes:
      - original (24)
      - delta (24)
      - acceleration (24)
      - rolling_std_5 (24)
      - rolling_max_5 (24)
    """
    N, T, F = X.shape
    
    # 1. Deltas
    deltas = np.zeros_like(X)
    deltas[:, 1:, :] = X[:, 1:, :] - X[:, :-1, :]
    
    # 2. Acceleration
    accel = np.zeros_like(X)
    accel[:, 1:, :] = deltas[:, 1:, :] - deltas[:, :-1, :]
    
    # 3 & 4. Rolling stats (window=5)
    roll_std = np.zeros_like(X)
    roll_max = np.zeros_like(X)
    
    for t in range(T):
        start_idx = max(0, t - 4)
        window_data = X[:, start_idx:t+1, :]
        roll_std[:, t, :] = np.std(window_data, axis=1)
        roll_max[:, t, :] = np.max(window_data, axis=1)
        
    return np.concatenate([X, deltas, accel, roll_std, roll_max], axis=-1)


def build_augmented_model() -> AttentionLSTM:
    return AttentionLSTM(
        input_size=AUG_INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        K=K,
        num_tactic_classes=NUM_CLASSES,
    )


def compute_class_weights_binary(y: np.ndarray) -> torch.Tensor:
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def compute_class_weights_mc(y: np.ndarray, n: int) -> torch.Tensor:
    counts = np.bincount(y.flatten(), minlength=n).astype(float)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    return torch.tensor(w / w.sum() * n, dtype=torch.float32)


def run_inference(model, X_aug, device, batch_size=512):
    """Return (N, K) risk probabilities."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, len(X_aug), batch_size):
            xb = torch.FloatTensor(X_aug[i:i+batch_size]).to(device)
            rp, _, _ = model(xb)
            chunks.append(rp.cpu().numpy())
    return np.vstack(chunks)


def train_augmented(X_tr, yr_tr, yt_tr, X_va, yr_va, yt_va,
                    device, checkpoint_path, max_epochs=60, patience=10):
    """Train augmented LSTM with early stopping on validation loss."""
    model = build_augmented_model().to(device)

    pos_w   = compute_class_weights_binary(yr_tr).to(device)
    tact_w  = compute_class_weights_mc(yt_tr, NUM_CLASSES).to(device)
    tact_ce = nn.CrossEntropyLoss(weight=tact_w)

    def risk_loss(probs, targets):
        return nn.functional.binary_cross_entropy(probs, targets)

    train_ds = TensorDataset(
        torch.FloatTensor(X_tr),
        torch.FloatTensor(yr_tr),
        torch.LongTensor(yt_tr),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_va),
        torch.FloatTensor(yr_va),
        torch.LongTensor(yt_va),
    )
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)

    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-5)

    best_val, best_ep, patience_cnt = float("inf"), 0, 0
    train_losses, val_losses = [], []

    for ep in range(1, max_epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yrb, ytb in train_loader:
            xb, yrb, ytb = xb.to(device), yrb.to(device), ytb.to(device)
            opt.zero_grad()
            rp, tl, _ = model(xb)
            b, Kk, C  = tl.shape
            loss = risk_loss(rp, yrb) + tact_ce(tl.reshape(b*Kk, C), ytb.reshape(b*Kk))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
        scheduler.step()
        tr_loss /= len(train_ds)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, yrb, ytb in val_loader:
                xb, yrb, ytb = xb.to(device), yrb.to(device), ytb.to(device)
                rp, tl, _ = model(xb)
                b, Kk, C  = tl.shape
                loss = risk_loss(rp, yrb) + tact_ce(tl.reshape(b*Kk, C), ytb.reshape(b*Kk))
                vl_loss += loss.item() * len(xb)
        vl_loss /= len(val_ds)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if ep % 5 == 0 or ep == 1:
            logger.info(f"Epoch {ep:3d} | train={tr_loss:.4f} | val={vl_loss:.4f}")

        if vl_loss < best_val - 1e-4:
            best_val, best_ep, patience_cnt = vl_loss, ep, 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                logger.info(f"Early stop @ epoch {ep}. Best: ep {best_ep} val={best_val:.4f}")
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model, {"train_loss": train_losses, "val_loss": val_losses,
                   "best_epoch": best_ep, "best_val_loss": best_val}


def compute_proactive_metrics(risk_probs, y_risk, threshold, window_sec):
    """3-way clean breakdown: proactive / reactive / missed."""
    proactive_lts, reactive_delays, missed = [], [], 0
    for i in range(len(y_risk)):
        onset_ks = np.where(y_risk[i] == 1)[0]
        if len(onset_ks) == 0:
            continue
        attack_t = (int(onset_ks[0]) + 1) * window_sec
        alert_ks = np.where(risk_probs[i] > threshold)[0]
        if len(alert_ks) == 0:
            missed += 1
            continue
        alert_t  = (int(alert_ks[0]) + 1) * window_sec
        lead     = attack_t - alert_t
        if lead > 0:
            proactive_lts.append(lead)
        else:
            reactive_delays.append(-lead)

    total = len(y_risk)
    return {
        "total": total,
        "proactive": {
            "count": len(proactive_lts),
            "rate": round(len(proactive_lts) / total, 4) if total else 0.0,
            "median_lead_sec": round(float(np.median(proactive_lts)), 1) if proactive_lts else 0.0,
            "mean_lead_sec":   round(float(np.mean(proactive_lts)), 1) if proactive_lts else 0.0,
        },
        "reactive": {
            "count": len(reactive_delays),
            "rate": round(len(reactive_delays) / total, 4) if total else 0.0,
        },
        "missed": {
            "count": missed,
            "rate": round(missed / total, 4) if total else 0.0,
        },
    }


def main():
    processed_dir = config.resolve_path("processed_data")
    models_dir    = config.resolve_path("models")
    reports_dir   = config.resolve_path("reports")
    sys_cfg       = config.system()
    window_sec    = sys_cfg["windowing"]["window_seconds"]
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Device: {device}  |  Augmented input size: {AUG_INPUT_SIZE}")

    # ── Load sequences ────────────────────────────────────────────────────────
    def load(tag):
        d = np.load(processed_dir / f"{tag}_sequences.npz")
        return d["X"], d["y_risk"], d["y_tactic"]

    X_tr, yr_tr, yt_tr = load("train")
    X_va, yr_va, yt_va = load("val")
    X_te, yr_te, yt_te = load("test")

    logger.info(f"Train: {len(X_tr):,} | Val: {len(X_va):,} | Test: {len(X_te):,}")

    # ── Augment with trend features ───────────────────────────────────────────
    logger.info("Adding full trend features (deltas, accel, rolling stats)...")
    X_tr_aug = add_trend_features(X_tr)
    X_va_aug = add_trend_features(X_va)
    X_te_aug = add_trend_features(X_te)
    logger.info(f"Augmented shape: {X_tr_aug.shape}  (was {X_tr.shape})")

    # ── Val-A / Val-B split ────────────────────────────────────────────────────
    N_va = len(X_va_aug)
    split_n = int(N_va * VAL_A_FRAC)
    X_va_a, yr_va_a = X_va_aug[:split_n], yr_va[:split_n]
    X_va_b, yr_va_b = X_va_aug[split_n:], yr_va[split_n:]

    # Proactive sequences in Val-B
    cb = yr_va_b[:, 0] == 0
    fa = yr_va_b[:, 1:].sum(axis=1) > 0
    pro_mask = cb & fa
    pro_idx  = np.where(pro_mask)[0]
    logger.info(f"Val-B proactive sequences: {pro_mask.sum():,}")

    # ── Train augmented LSTM ──────────────────────────────────────────────────
    checkpoint_path = models_dir / "lstm_augmented_checkpoint.pt"
    logger.info("Training augmented LSTM (lr=5e-4, weight_decay=1e-4, cosine LR)...")
    model, curves = train_augmented(
        X_tr_aug, yr_tr, yt_tr,
        X_va_a,   yr_va_a, yt_va[:split_n],
        device, checkpoint_path,
        max_epochs=60, patience=10
    )

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    rp_te   = run_inference(model, X_te_aug, device)
    k1_preds = (rp_te[:, 0] > THRESHOLD).astype(int)
    k1_f1    = float(f1_score(yr_te[:, 0], k1_preds, average="macro", zero_division=0))

    k_f1s = {}
    for k in range(K):
        p = (rp_te[:, k] > THRESHOLD).astype(int)
        k_f1s[f"k{k+1}"] = round(float(f1_score(yr_te[:, k], p, average="macro", zero_division=0)), 4)

    print(f"\n{SEP}")
    print("PHASE 7.4 — ADVANCED AUGMENTED LSTM RESULTS")
    print(SEP)
    print(f"\nBest epoch: {curves['best_epoch']}  |  Best val loss: {curves['best_val_loss']:.4f}")
    print(f"\nTest k=1 risk Macro-F1 (threshold={THRESHOLD:.2f}): {k1_f1:.4f}")
    print(f"Per-horizon risk F1: {k_f1s}")

    # ── Proactive evaluation on Val-B ─────────────────────────────────────────
    logger.info("Computing proactive metrics on Val-B...")
    rp_b_pro = run_inference(model, X_va_aug[split_n:][pro_idx], device)
    yr_b_pro = yr_va_b[pro_idx]
    det = compute_proactive_metrics(rp_b_pro, yr_b_pro, THRESHOLD, window_sec)

    p, r, m = det["proactive"], det["reactive"], det["missed"]
    print(f"\n{SEP}")
    print(f"PROACTIVE EVALUATION (Val-B, threshold={THRESHOLD:.2f})")
    print(SEP)
    print(f"  Eligible attack sequences:  {det['total']:,}")
    print(f"\n  PROACTIVE (before onset):  {p['count']:,}  ({p['rate']:.1%})")
    print(f"    Median lead:              {p['median_lead_sec']:+.1f}s")
    print(f"    Mean lead:                {p['mean_lead_sec']:+.1f}s")
    print(f"\n  REACTIVE (at/after onset): {r['count']:,}  ({r['rate']:.1%})")
    print(f"\n  MISSED:                    {m['count']:,}  ({m['rate']:.1%})")

    # ── Compare with Phase 7.1 baseline ──────────────────────────────────────
    print(f"\n{'─'*70}")
    print("COMPARISON: Phase 7.1 (original) vs Phase 7.4 (advanced trends augmented)")
    print(f"{'─'*70}")
    print(f"  Original LSTM  PDR={0.200:.1%}  Median lead=+25s  (threshold=0.30)")
    print(f"  Advanced LSTM PDR={p['rate']:.1%}  Median lead={p['median_lead_sec']:+.0f}s")
    improvement = (p['rate'] - 0.200) * 100
    print(f"  PDR change: {improvement:+.1f}pp")

    # ── Save ──────────────────────────────────────────────────────────────────
    report = {
        "experiment": "Phase 7.4 — Advanced Delta-Augmented LSTM",
        "augmented_input_size": AUG_INPUT_SIZE,
        "threshold": THRESHOLD,
        "best_epoch": curves["best_epoch"],
        "best_val_loss": round(curves["best_val_loss"], 6),
        "test_k1_macro_f1": round(k1_f1, 4),
        "test_per_horizon_f1": k_f1s,
        "val_b_proactive_metrics": det,
        "baseline_comparison": {
            "original_pdr": 0.200,
            "augmented_pdr": p['rate'],
            "pdr_improvement_pp": round(improvement, 2),
        },
    }
    out = reports_dir / "phase7_4_augmented_metrics.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Phase 7.4 advanced report saved to {out}")


if __name__ == "__main__":
    main()
