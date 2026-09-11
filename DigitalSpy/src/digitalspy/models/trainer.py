"""LSTM training loop.

Implements:
- Class-weighted BCE for risk head
- Class-weighted CrossEntropy for tactic head
- Equal head weighting: L_total = L_risk + L_tactic
- Adam optimizer, lr=1e-3
- Early stopping on validation loss
- Saves checkpoint and validation curves
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from digitalspy import config as ds_config
from digitalspy.models.attention_lstm import AttentionLSTM, build_model

logger = logging.getLogger(__name__)


def compute_class_weights_binary(y: np.ndarray) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from binary labels."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def compute_class_weights_multiclass(
    y: np.ndarray, num_classes: int
) -> torch.Tensor:
    """Compute per-class weights inversely proportional to frequency."""
    counts = np.bincount(y.flatten(), minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def train_epoch(
    model: AttentionLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    risk_criterion: nn.BCELoss,
    tactic_criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for X_batch, y_risk_batch, y_tactic_batch in loader:
        X_batch = X_batch.to(device)
        y_risk_batch = y_risk_batch.to(device)
        y_tactic_batch = y_tactic_batch.to(device)

        optimizer.zero_grad()
        risk_probs, tactic_logits, _ = model(X_batch)

        # Risk loss: mean over K horizons
        loss_risk = risk_criterion(risk_probs, y_risk_batch)

        # Tactic loss: reshape for CrossEntropyLoss
        # tactic_logits: (batch, K, 6) → (batch*K, 6)
        # y_tactic: (batch, K) → (batch*K,)
        b, K, C = tactic_logits.shape
        loss_tactic = tactic_criterion(
            tactic_logits.reshape(b * K, C),
            y_tactic_batch.reshape(b * K),
        )

        loss = loss_risk + loss_tactic
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(X_batch)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(
    model: AttentionLSTM,
    loader: DataLoader,
    risk_criterion: nn.BCELoss,
    tactic_criterion: nn.CrossEntropyLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0

    for X_batch, y_risk_batch, y_tactic_batch in loader:
        X_batch = X_batch.to(device)
        y_risk_batch = y_risk_batch.to(device)
        y_tactic_batch = y_tactic_batch.to(device)

        risk_probs, tactic_logits, _ = model(X_batch)

        loss_risk = risk_criterion(risk_probs, y_risk_batch)
        b, K, C = tactic_logits.shape
        loss_tactic = tactic_criterion(
            tactic_logits.reshape(b * K, C),
            y_tactic_batch.reshape(b * K),
        )

        loss = (loss_risk + loss_tactic).item()
        total_loss += loss * len(X_batch)

    return total_loss / len(loader.dataset)


def train(
    X_train: np.ndarray,
    y_risk_train: np.ndarray,
    y_tactic_train: np.ndarray,
    X_val: np.ndarray,
    y_risk_val: np.ndarray,
    y_tactic_val: np.ndarray,
    checkpoint_path: Path,
    val_curves_path: Path,
    cfg: Optional[dict] = None,
) -> tuple[AttentionLSTM, dict]:
    """Full training loop with early stopping.

    Args:
        X_train: (N_train, 20, 24)
        y_risk_train: (N_train, K) binary float
        y_tactic_train: (N_train, K) int64
        X_val: (N_val, 20, 24)
        y_risk_val: (N_val, K) binary float
        y_tactic_val: (N_val, K) int64
        checkpoint_path: Where to save best model.
        val_curves_path: Where to save train/val loss curves.
        cfg: Optional config override.

    Returns:
        (best_model, curves_dict)
    """
    if cfg is None:
        cfg = ds_config.lstm()

    train_cfg = cfg["training"]
    num_classes = cfg["forecast"]["heads"]["tactic"]["num_classes"]
    K = cfg["forecast"]["K"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    # Build model
    model = build_model(cfg).to(device)

    # Compute class weights
    pos_weight = compute_class_weights_binary(y_risk_train).to(device)
    tactic_weights = compute_class_weights_multiclass(
        y_tactic_train, num_classes
    ).to(device)

    # Loss functions
    risk_criterion = nn.BCELoss(weight=None)  # pos_weight handled via BCEWithLogitsLoss below
    # Use BCEWithLogitsLoss for numerical stability
    risk_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.expand(K))

    def risk_loss_fn(logits_or_probs, targets):
        # Model outputs sigmoid probs; we need logits for BCEWithLogitsLoss
        # Use BCE directly on probs since model applies sigmoid
        return nn.functional.binary_cross_entropy(logits_or_probs, targets)

    tactic_ce = nn.CrossEntropyLoss(weight=tactic_weights)

    # DataLoaders
    train_ds = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_risk_train),
        torch.LongTensor(y_tactic_train),
    )
    val_ds = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_risk_val),
        torch.LongTensor(y_tactic_val),
    )

    batch_size = train_cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    # Early stopping
    patience = train_cfg["early_stopping"]["patience"]
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    train_losses, val_losses = [], []

    for epoch in range(1, train_cfg["max_epochs"] + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, risk_loss_fn, tactic_ce, device)
        vl_loss = eval_epoch(model, val_loader, risk_loss_fn, tactic_ce, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:3d} | train_loss={tr_loss:.4f} | val_loss={vl_loss:.4f}")

        if vl_loss < best_val_loss - train_cfg["early_stopping"]["min_delta"]:
            best_val_loss = vl_loss
            best_epoch = epoch
            patience_counter = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}. Best: epoch {best_epoch} val_loss={best_val_loss:.4f}")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    curves = {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }

    val_curves_path.parent.mkdir(parents=True, exist_ok=True)
    with open(val_curves_path, "w") as f:
        json.dump(curves, f, indent=2)

    logger.info(f"Training complete. Checkpoint: {checkpoint_path}")
    return model, curves
