"""Logistic Regression baseline models.

Two models per IMPLEMENTATION.md §10:
  1. Current detection:   S_t → Y_t   (binary risk + 6-class tactic)
  2. One-step forecasting baseline: S_t → Y_{t+1}

Metrics: Precision, Recall, Macro-F1, FPR.
Accuracy is NOT the headline metric.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelBinarizer

logger = logging.getLogger(__name__)


def compute_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute False Positive Rate for binary classification."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp) / float(fp + tn + 1e-9)


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    label: str = "",
) -> dict:
    """Compute binary classification metrics (Precision, Recall, Macro-F1, FPR)."""
    metrics = {
        "label": label,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "fpr": compute_fpr(y_true, y_pred),
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
    }
    return metrics


def evaluate_multiclass(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    label: str = "",
) -> dict:
    """Compute multi-class metrics: Macro-F1, per-class Precision/Recall."""
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    return {
        "label": label,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in class_names
            if cls in report
        },
        "n_samples": int(len(y_true)),
    }


class LogisticBaseline:
    """Wrapper for scikit-learn LogisticRegression with DigitalSpy conventions."""

    def __init__(self, task: str = "binary", class_names: Optional[list[str]] = None):
        """
        Args:
            task: 'binary' for risk head, 'multiclass' for tactic head.
            class_names: For multiclass; list of Z_t bucket names.
        """
        if task not in ("binary", "multiclass"):
            raise ValueError(f"task must be 'binary' or 'multiclass', got '{task}'")
        self.task = task
        self.class_names = class_names or []
        self._model: Optional[LogisticRegression] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit logistic regression with class-weighted loss."""
        solver = "lbfgs" if self.task == "binary" else "lbfgs"
        self._model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver=solver,
            random_state=42,
        )
        self._model.fit(X_train, y_train)
        logger.info(f"LogisticBaseline[{self.task}] fitted on {len(X_train):,} samples.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.predict_proba(X)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray, label: str = "") -> dict:
        """Evaluate and return metrics dict."""
        y_pred = self.predict(X)
        if self.task == "binary":
            return evaluate_binary(y_true, y_pred, label=label)
        else:
            return evaluate_multiclass(y_true, y_pred, self.class_names, label=label)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path)
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: Path, task: str = "binary",
             class_names: Optional[list[str]] = None) -> "LogisticBaseline":
        obj = cls(task=task, class_names=class_names)
        obj._model = joblib.load(path)
        return obj
