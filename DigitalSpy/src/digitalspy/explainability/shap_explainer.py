"""SHAP feature attribution for DigitalSpy models.

Provides SHAP-based explanations for:
  - Logistic Regression (LinearExplainer)
  - Attention-LSTM (DeepExplainer or GradientExplainer)

XAI language rules (IMPLEMENTATION.md §21):
  ✓ "Most influential features"
  ✓ "Feature evidence associated with forecast"
  ✗ "caused by"
  ✗ "proves causality"
  ✗ "exact cause"
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def explain_logistic(
    model,
    X_background: np.ndarray,
    X_instance: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
) -> dict:
    """SHAP explanation for logistic regression.

    Args:
        model: Fitted LogisticBaseline or sklearn LogisticRegression.
        X_background: Background dataset (subset of training data).
        X_instance: Single instance to explain, shape (1, 24) or (24,).
        feature_names: List of 24 feature names.
        top_k: Number of top features to return.

    Returns:
        dict with top_features and feature_contributions.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed. Returning empty explanation.")
        return {"top_features": [], "feature_contributions": {}}

    if hasattr(model, "_model"):
        sklearn_model = model._model
    else:
        sklearn_model = model

    explainer = shap.LinearExplainer(sklearn_model, X_background)
    shap_values = explainer.shap_values(X_instance)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # positive class for binary
    else:
        shap_vals = shap_values

    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    # Build feature contribution dict
    contributions = {
        name: float(val)
        for name, val in zip(feature_names, shap_vals)
    }

    # Top-K by absolute value
    sorted_feats = sorted(
        contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )
    top_features = [f for f, _ in sorted_feats[:top_k]]

    return {
        "top_features": top_features,
        "feature_contributions": {f: contributions[f] for f in top_features},
        "xai_note": "SHAP values indicate feature influence, not causal proof.",
    }


def explain_lstm(
    model,
    X_background: np.ndarray,
    X_instance: np.ndarray,
    feature_names: list[str],
    top_k: int = 10,
    device: str = "cpu",
) -> dict:
    """SHAP explanation for Attention-LSTM (GradientExplainer).

    Args:
        model: Fitted AttentionLSTM (PyTorch).
        X_background: Background sample, shape (N_bg, 20, 24).
        X_instance: Single instance, shape (1, 20, 24) or (20, 24).
        feature_names: List of 24 feature names.
        top_k: Number of top features to return.
        device: 'cpu' or 'cuda'.

    Returns:
        dict with top_features and feature_contributions (averaged over time).
    """
    try:
        import shap
        import torch
    except ImportError:
        logger.warning("shap/torch not available. Returning empty explanation.")
        return {"top_features": [], "feature_contributions": {}}

    model.eval()

    import torch
    bg = torch.FloatTensor(X_background[:50]).to(device)  # limit background size

    if X_instance.ndim == 2:
        X_instance = X_instance[np.newaxis]

    inst = torch.FloatTensor(X_instance).to(device)

    def model_fn(x):
        """Return risk prob for GradientExplainer."""
        with torch.no_grad():
            risk, _, _ = model(x)
        return risk[:, 0:1]  # k=1 risk only

    try:
        explainer = shap.GradientExplainer(model_fn, bg)
        shap_vals = explainer.shap_values(inst)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        # shap_vals: (1, 20, 24) → average over time axis
        mean_shap = np.abs(shap_vals[0]).mean(axis=0)  # (24,)

        contributions = {
            name: float(val)
            for name, val in zip(feature_names, mean_shap)
        }

        sorted_feats = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )
        top_features = [f for f, _ in sorted_feats[:top_k]]

        return {
            "top_features": top_features,
            "feature_contributions": {f: contributions[f] for f in top_features},
            "xai_note": "SHAP values averaged over 20-window history. Indicates model focus, not causality.",
        }
    except Exception as e:
        logger.warning(f"LSTM SHAP failed: {e}. Returning empty.")
        return {"top_features": [], "feature_contributions": {}, "error": str(e)}
