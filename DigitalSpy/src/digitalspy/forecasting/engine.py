"""Forecast engine — direct multi-horizon prediction.

Input:  H_t = S_{t-19:t}  shape (20, 24)
Output: JSON with risk[k=1..5], tactic[k=1..5], attention_weights, current_state

This is DIRECT multi-horizon: the model outputs all K horizons in one forward
pass. It does NOT recursively feed predictions back into the model (Phase-2).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from digitalspy import config as ds_config
from digitalspy.models.attention_lstm import AttentionLSTM, build_model
from digitalspy.states.labels import index_to_label

logger = logging.getLogger(__name__)


class ForecastEngine:
    """Wraps the Attention-LSTM for inference.

    Produces structured forecast JSON per IMPLEMENTATION.md §6 and §25.
    The agent may read but never modify the probabilities in this output.
    """

    def __init__(self, checkpoint_path: Path, cfg: Optional[dict] = None):
        if cfg is None:
            cfg = ds_config.lstm()
        self.cfg = cfg
        self.K = cfg["forecast"]["K"]
        self.num_classes = cfg["forecast"]["heads"]["tactic"]["num_classes"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = build_model(cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(checkpoint_path, map_location=self.device)
        )
        self.model.eval()
        logger.info(f"ForecastEngine loaded from {checkpoint_path}")

    def predict(
        self,
        history: np.ndarray,
        current_z_t: Optional[str] = None,
    ) -> dict:
        """Run inference on a 20×24 history and return structured forecast.

        Args:
            history: (20, 24) float32 array of scaled S_t vectors.
            current_z_t: Optional current Z_t label (for display).

        Returns:
            dict with keys:
                current_state: str
                risk: [p1, p2, p3, p4, p5]          — floats [0,1]
                tactic_probs: [[6 probs], ...]       — one per horizon
                tactic_labels: [label, ...]          — argmax tactic per horizon
                attention_weights: [a1..a20]         — sum to 1
                horizon_seconds: [10, 20, 30, 40, 50]
        """
        if history.shape != (20, 24):
            raise ValueError(f"History must be (20, 24), got {history.shape}")

        x = torch.FloatTensor(history).unsqueeze(0).to(self.device)  # (1, 20, 24)

        with torch.no_grad():
            risk_probs, tactic_logits, alpha = self.model(x)

        risk = risk_probs.squeeze(0).cpu().numpy().tolist()             # [K]
        tactic_raw = tactic_logits.squeeze(0).cpu().numpy()            # (K, 6)
        tactic_probs = torch.softmax(
            torch.tensor(tactic_raw), dim=-1
        ).numpy().tolist()                                               # (K, 6)
        attn = alpha.squeeze(0).cpu().numpy().tolist()                  # [20]

        tactic_labels = [
            index_to_label(int(np.argmax(t))) for t in tactic_probs
        ]

        # Validate probability ranges (red flag §29)
        for p in risk:
            assert 0.0 <= p <= 1.0, f"Risk probability out of range: {p}"

        return {
            "current_state": current_z_t or "UNKNOWN",
            "risk": [round(p, 4) for p in risk],
            "tactic_probs": [[round(p, 4) for p in row] for row in tactic_probs],
            "tactic_labels": tactic_labels,
            "attention_weights": [round(a, 4) for a in attn],
            "horizon_seconds": [10 * (k + 1) for k in range(self.K)],
        }

    def predict_batch(self, histories: np.ndarray) -> list[dict]:
        """Run batch inference. histories: (N, 20, 24)."""
        results = []
        for i in range(len(histories)):
            results.append(self.predict(histories[i]))
        return results
