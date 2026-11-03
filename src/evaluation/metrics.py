"""Classification metrics for held-out frame/image evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(labels: list[int] | np.ndarray, probabilities: list[float] | np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Calculate thresholded binary metrics and ROC-AUC."""
    y_true = np.asarray(labels, dtype=int)
    y_prob = np.asarray(probabilities, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = None
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = None
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1], target_names=["real", "fake"], output_dict=True, zero_division=0
        ),
    }
