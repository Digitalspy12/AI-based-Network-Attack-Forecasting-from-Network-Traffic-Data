"""Attention-LSTM model for multi-horizon network attack forecasting.

Architecture (frozen, IMPLEMENTATION.md §12–14):
    Input:  (batch, 20, 24)
    LSTM:   1 layer, hidden=64, dropout=0.20
    Attn:   Additive (Bahdanau-style) over 20 hidden states → context vector
    Risk:   5 sigmoid outputs (one per horizon k=1..5)
    Tactic: 5 × softmax(6) outputs (one 6-class distribution per horizon)

Loss:
    L_risk   = sum_k w(y_k) · BCE(y_k, ŷ_k)        [weighted binary CE]
    L_tactic = sum_k CE_weighted(y_k, ŷ_k)           [weighted cross-entropy]
    L_total  = L_risk + L_tactic                       [equal weighting]

Frozen parameters:
    input_size=24, sequence_length=20, hidden_size=64,
    num_layers=1, dropout=0.20, K=5, num_tactic_classes=6
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Frozen architecture constants (do NOT change without §30 change control)
_INPUT_SIZE = 24
_SEQ_LEN = 20
_HIDDEN_SIZE = 64
_NUM_LAYERS = 1
_DROPOUT = 0.20
_K = 5
_NUM_TACTIC_CLASSES = 6


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over LSTM hidden states.

    e_i = v^T · tanh(W_h · h_i + b_h)
    α_i = softmax(e_i)
    c   = Σ α_i · h_i
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            context: (batch, hidden_size)
            alpha: (batch, seq_len) — attention weights (stored for inference)
        """
        # e_i = v^T tanh(W_h h_i + b_h)
        energy = self.v(torch.tanh(self.W_h(hidden_states)))  # (batch, seq_len, 1)
        energy = energy.squeeze(-1)                             # (batch, seq_len)

        alpha = F.softmax(energy, dim=1)                        # (batch, seq_len)
        # c = Σ α_i h_i
        context = torch.bmm(alpha.unsqueeze(1), hidden_states)  # (batch, 1, hidden)
        context = context.squeeze(1)                            # (batch, hidden)

        return context, alpha


class AttentionLSTM(nn.Module):
    """One-layer LSTM with additive attention and dual forecast heads.

    Dual output heads (K outputs each):
        risk_head:   K sigmoid probabilities — binary risk per horizon
        tactic_head: K × 6 log-softmax distributions — tactic per horizon
    """

    def __init__(
        self,
        input_size: int = _INPUT_SIZE,
        hidden_size: int = _HIDDEN_SIZE,
        num_layers: int = _NUM_LAYERS,
        dropout: float = _DROPOUT,
        K: int = _K,
        num_tactic_classes: int = _NUM_TACTIC_CLASSES,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.K = K
        self.num_tactic_classes = num_tactic_classes

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Dropout applied after LSTM (single layer)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = AdditiveAttention(hidden_size)

        # Risk head: K binary outputs
        self.risk_head = nn.Linear(hidden_size, K)

        # Tactic head: K × num_tactic_classes outputs
        self.tactic_head = nn.Linear(hidden_size, K * num_tactic_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_size) = (batch, 20, 24)

        Returns:
            risk_probs:   (batch, K) — sigmoid risk probabilities
            tactic_logits: (batch, K, num_tactic_classes) — raw logits for CE loss
            alpha:        (batch, seq_len) — attention weights
        """
        # LSTM: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)

        # Attention over all 20 hidden states
        context, alpha = self.attention(lstm_out)  # context: (batch, hidden)

        # Risk head
        risk_logits = self.risk_head(context)       # (batch, K)
        risk_probs = torch.sigmoid(risk_logits)

        # Tactic head
        tactic_raw = self.tactic_head(context)      # (batch, K * num_classes)
        tactic_logits = tactic_raw.view(
            -1, self.K, self.num_tactic_classes
        )                                            # (batch, K, num_classes)

        return risk_probs, tactic_logits, alpha

    def get_attention_weights(self, x: torch.Tensor) -> np.ndarray:
        """Extract attention weights for a single sample (inference).

        Args:
            x: (1, seq_len, input_size) or (seq_len, input_size)

        Returns:
            alpha: (seq_len,) numpy array
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 2:
                x = x.unsqueeze(0)
            _, _, alpha = self.forward(x)
        return alpha.squeeze(0).cpu().numpy()


def build_model(cfg: Optional[dict] = None) -> AttentionLSTM:
    """Instantiate AttentionLSTM from lstm.yaml config.

    Args:
        cfg: Optional override dict. If None, loads from config file.

    Returns:
        AttentionLSTM instance.
    """
    from digitalspy import config as ds_config
    if cfg is None:
        cfg = ds_config.lstm()

    arch = cfg["architecture"]
    return AttentionLSTM(
        input_size=arch["input_size"],
        hidden_size=arch["hidden_size"],
        num_layers=arch["num_layers"],
        dropout=arch["dropout"],
        K=cfg["forecast"]["K"],
        num_tactic_classes=cfg["forecast"]["heads"]["tactic"]["num_classes"],
    )
