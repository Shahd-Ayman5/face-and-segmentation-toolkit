from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float          # macro-averaged
    recall: float             # macro-averaged
    f1: float                 # macro-averaged
    confusion_matrix: np.ndarray
    labels: np.ndarray        # unique class labels in order


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """
    Compute accuracy, macro-averaged precision/recall/F1, and confusion matrix.

    Parameters
    ----------
    y_true : 1-D array of ground-truth subject IDs
    y_pred : 1-D array of predicted subject IDs (same length)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    labels = np.unique(y_true)
    n = len(labels)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    # --- confusion matrix ---
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[label_to_idx[t], label_to_idx[p]] += 1

    # --- accuracy ---
    accuracy = float(np.trace(cm)) / float(len(y_true))

    # --- per-class precision / recall / F1 (macro average) ---
    precisions, recalls, f1s = [], [], []
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        f1=float(np.mean(f1s)),
        confusion_matrix=cm,
        labels=labels,
    )