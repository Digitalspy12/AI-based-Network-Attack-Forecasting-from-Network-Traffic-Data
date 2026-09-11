"""Attention weight visualisation utilities.

Extracts and formats attention weights from the Attention-LSTM for display
in the Streamlit SOC UI.

Interpretation note (IMPLEMENTATION.md §13):
  "Attention identifies historical windows the model weighted more heavily.
   It is evidence about model focus, not causal proof."
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def format_attention_timeline(
    attention_weights: list[float],
    window_starts: list[str] | None = None,
    history_length: int = 20,
) -> dict:
    """Format attention weights for display.

    Args:
        attention_weights: List of 20 attention weights (sum ~1).
        window_starts: Optional list of 20 timestamp strings.
        history_length: Expected length (= 20).

    Returns:
        dict with weights, timestep labels, peak timestep, and note.
    """
    weights = np.array(attention_weights, dtype=float)

    if len(weights) != history_length:
        raise ValueError(
            f"Expected {history_length} attention weights, got {len(weights)}"
        )

    # Normalise (should already sum to 1 from softmax, but be safe)
    if weights.sum() > 0:
        weights = weights / weights.sum()

    timestep_labels = (
        window_starts
        if window_starts and len(window_starts) == history_length
        else [f"t-{history_length - 1 - i}" for i in range(history_length)]
    )

    peak_idx = int(np.argmax(weights))
    peak_label = timestep_labels[peak_idx]

    return {
        "attention_weights": weights.tolist(),
        "timestep_labels": timestep_labels,
        "peak_timestep": peak_label,
        "peak_weight": float(weights[peak_idx]),
        "top3_timesteps": [
            timestep_labels[i]
            for i in np.argsort(weights)[::-1][:3]
        ],
        "note": (
            "High attention on a timestep means the model weighted that "
            "historical window more heavily. This is model evidence, "
            "not proof of causation."
        ),
    }


def attention_to_dataframe(
    attention_weights: list[float],
    window_starts: list[str] | None = None,
) -> pd.DataFrame:
    """Convert attention weights to a DataFrame for Plotly visualisation."""
    n = len(attention_weights)
    labels = (
        window_starts
        if window_starts and len(window_starts) == n
        else [f"t-{n - 1 - i}" for i in range(n)]
    )
    return pd.DataFrame({
        "timestep": labels,
        "attention_weight": attention_weights,
        "timestep_idx": list(range(n)),
    })
