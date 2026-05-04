from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from core.face.pca_model import PCA
from core.face.process_orl import prepare_data
from core.face.metrics import ClassificationMetrics, compute_metrics


@dataclass
class EvaluationResult:
    """All data produced by a full test-set evaluation run."""
    metrics: ClassificationMetrics

    # ROC data (one-vs-rest, macro-averaged)
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float

    # Raw predictions
    y_true: np.ndarray
    y_pred: np.ndarray
    distances: np.ndarray


def _compute_roc_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_subjects: np.ndarray,
    train_projections: np.ndarray,
    pca: PCA,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    One-vs-rest ROC averaged over all subjects.
    For each subject, the score for a test sample is:
      score = - (mean distance to that subject's training images)
    No normalization — raw distances give more spread.
    """
    labels = np.unique(y_true)
    all_fpr = np.linspace(0, 1, 200)
    tprs = []

    # Project all test images once
    test_projections = np.vstack([
        pca.transform(vec.reshape(1, -1)) for vec in X_test
    ])  # (n_test, d)

    for subject in labels:
        binary_true = (y_true == subject).astype(int)  # 1 if this subject, 0 otherwise

        subject_mask = train_subjects == subject
        subject_projs = train_projections[subject_mask]  # (k, d)

        # For each test image: compute mean distance to this subject's training images
        # Negative so that closer (smaller distance) = higher score
        scores = np.array([
            -float(np.mean(np.linalg.norm(subject_projs - proj, axis=1)))
            for proj in test_projections
        ])

        # Sort thresholds from high to low
        sorted_scores = np.sort(scores)[::-1]

        fprs_sub = [0.0]
        tprs_sub = [0.0]

        for thr in sorted_scores:
            predicted_pos = (scores >= thr)
            tp = int(np.logical_and(predicted_pos,  binary_true).sum())
            fp = int(np.logical_and(predicted_pos, ~binary_true.astype(bool)).sum())
            fn = int(np.logical_and(~predicted_pos, binary_true.astype(bool)).sum())
            tn = int(np.logical_and(~predicted_pos, ~binary_true.astype(bool)).sum())

            tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

            fprs_sub.append(fpr_val)
            tprs_sub.append(tpr_val)

        fprs_sub.append(1.0)
        tprs_sub.append(1.0)

        # Sort by fpr for interpolation
        fprs_sub = np.array(fprs_sub)
        tprs_sub = np.array(tprs_sub)
        sort_idx = np.argsort(fprs_sub)
        fprs_sub = fprs_sub[sort_idx]
        tprs_sub = tprs_sub[sort_idx]

        interp_tpr = np.interp(all_fpr, fprs_sub, tprs_sub)
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0]  = 0.0
    mean_tpr[-1] = 1.0
    auc = float(np.trapezoid(mean_tpr, all_fpr))
    return all_fpr, mean_tpr, auc


def evaluate_model(model_path: str | Path) -> EvaluationResult:
    """
    Load the saved PCA model, run it on the held-out test split,
    and return full metrics + ROC data.
    """
    model_path = Path(model_path)
    pca = PCA.load(str(model_path))

    X_train, X_test, y_train, y_test, train_paths, test_paths = prepare_data()

    train_projections = pca.train_projections

    # --- predict each test sample ---
    y_pred = []
    distances = []
    for vec in X_test:
        proj  = pca.transform(vec.reshape(1, -1))
        dists = np.linalg.norm(train_projections - proj, axis=1)
        idx   = int(np.argmin(dists))
        y_pred.append(int(y_train[idx]))
        distances.append(float(dists[idx]))

    y_pred    = np.array(y_pred)
    distances = np.array(distances)

    metrics = compute_metrics(y_test, y_pred)

    fpr, tpr, auc = _compute_roc_macro(
        y_test, y_pred, y_train, train_projections, pca, X_test
    )

    return EvaluationResult(
        metrics=metrics,
        fpr=fpr,
        tpr=tpr,
        auc=auc,
        y_true=y_test,
        y_pred=y_pred,
        distances=distances,
    )