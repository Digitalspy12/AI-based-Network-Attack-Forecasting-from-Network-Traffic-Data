"""Markov World Model — first-order discrete state transitions.

Estimates P_{ij} = P(Z_{t+1} = j | Z_t = i) from training transitions.
Implements K-step rollforward: p_{t+k} = p_t · P^k

The Markov model is the project's most transparent implementation of
state-transition dynamics (IMPLEMENTATION.md §11).

Output:
    models/markov_matrix.npy
    reports/markov_transition_matrix.csv
    reports/markov_metrics.json
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from digitalspy import config
from digitalspy.states.labels import label_to_index, index_to_label

logger = logging.getLogger(__name__)


class MarkovWorldModel:
    """First-order 6-state discrete Markov model."""

    def __init__(self, num_states: int = 6, alpha: float = 1.0):
        """
        Args:
            num_states: Number of Z_t states (frozen = 6).
            alpha: Laplace smoothing parameter.
        """
        self.num_states = num_states
        self.alpha = alpha
        self.P: np.ndarray = np.zeros((num_states, num_states))  # transition matrix
        self._counts: np.ndarray = np.zeros((num_states, num_states))
        self._fitted = False

    def fit(self, z_sequence: np.ndarray) -> "MarkovWorldModel":
        """Estimate transition matrix from a sequence of Z_t indices.

        Only consecutive pairs from training data are used.
        Rows that are not consecutive (different hosts or time gaps)
        must be excluded by the caller.

        Args:
            z_sequence: 1D array of Z_t integer indices (training only).

        Returns:
            self
        """
        counts = np.full(
            (self.num_states, self.num_states),
            fill_value=self.alpha,  # Laplace smoothing
            dtype=np.float64,
        )

        for i in range(len(z_sequence) - 1):
            zi = int(z_sequence[i])
            zj = int(z_sequence[i + 1])
            if 0 <= zi < self.num_states and 0 <= zj < self.num_states:
                counts[zi, zj] += 1.0

        self._counts = counts
        # Normalize rows to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        self.P = counts / np.maximum(row_sums, 1e-12)
        self._fitted = True

        logger.info(f"Markov model fitted. Min row sum: {self.P.sum(axis=1).min():.6f}")
        return self

    def predict_k_step(self, z0: int, K: int) -> np.ndarray:
        """Compute K-step marginal distributions from initial state z0.

        p_{t+k} = e_{z0} · P^k

        Args:
            z0: Initial discrete state index.
            K: Number of steps ahead.

        Returns:
            (K, num_states) array of marginal probability distributions.
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        p = np.zeros(self.num_states)
        p[z0] = 1.0

        results = []
        P_k = np.eye(self.num_states)
        for _ in range(K):
            P_k = P_k @ self.P
            results.append(p @ P_k)

        return np.stack(results)  # (K, num_states)

    def predict_tactic_sequence(
        self,
        z_sequence: np.ndarray,
        K: int = 5,
    ) -> np.ndarray:
        """For a sequence of Z_t indices, predict K-step most likely tactic.

        Args:
            z_sequence: 1D array of current Z_t indices (one per window).
            K: Forecast horizon.

        Returns:
            (len(z_sequence), K) array of predicted Z_t indices (argmax).
        """
        preds = []
        for z0 in z_sequence:
            dist = self.predict_k_step(int(z0), K)  # (K, 6)
            preds.append(np.argmax(dist, axis=1))   # (K,)
        return np.stack(preds)  # (N, K)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.P)
        logger.info(f"Saved Markov matrix to {path}")

    @classmethod
    def load(cls, path: Path, num_states: int = 6) -> "MarkovWorldModel":
        obj = cls(num_states=num_states)
        obj.P = np.load(path)
        obj._fitted = True
        return obj

    def to_dataframe(self) -> pd.DataFrame:
        """Return transition matrix as a labeled DataFrame."""
        labels = config.labels()["states"]
        return pd.DataFrame(self.P, index=labels, columns=labels)

    def validate(self) -> None:
        """Verify rows approximately sum to 1 (red flag check §29)."""
        row_sums = self.P.sum(axis=1)
        for i, s in enumerate(row_sums):
            if not np.isclose(s, 1.0, atol=1e-4):
                raise ValueError(
                    f"Markov matrix row {i} sums to {s:.6f}, not 1.0. "
                    "Check for data leakage or computation error."
                )
        logger.info("Markov validation passed: all rows sum to ~1.0")
