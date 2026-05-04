from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from core.face.pca_model import PCA
from core.face.process_orl import prepare_data
from core.face.metrics import ClassificationMetrics, compute_metrics


@dataclass
class EvaluationResult:
    """All data produced by a full test-set evaluation run."""
    metrics: ClassificationMetrics

    # ROC data (one-vs-rest, macro-averaged)
    fpr: np.ndarray          # false-positive-rate thresholds
    tpr: np.ndarray          # true-positive-rate thresholds
    auc: float

    # Raw predictions (for debugging / further plots)
    y_true: np.ndarray
    y_pred: np.ndarray
    distances: np.ndarray    # nearest-neighbour distance for each test sample


def _compute_roc_macro(
    y_true: np.ndarray,
    distances: np.ndarray,
    train_subjects: np.ndarray,
    train_projections: np.ndarray,
    pca: PCA,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    One-vs-rest ROC averaged over all subjects.
    We use the minimum distance to any training image of the *correct* class
    as the positive score (lower distance = more confident match).
    We flip to similarity = -distance so higher = more positive.
    """
    labels = np.unique(y_true)
    all_fpr = np.linspace(0, 1, 200)
    tprs = []

    for subject in labels:
        # binary: is this sample the current subject?
        binary_true = (y_true == subject).astype(int)

        # score: -min_distance to any training sample of this subject
        subject_mask = train_subjects == subject
        subject_projs = train_projections[subject_mask]  # shape (k, d)

        scores = []
        for vec in X_test:
            proj = pca.transform(vec.reshape(1, -1))           # (1, d)
            dists = np.linalg.norm(subject_projs - proj, axis=1)
            scores.append(-float(dists.min()))                 # higher = more likely this subject
        scores = np.array(scores)

        # compute ROC for this subject
        thresholds = np.unique(scores)
        fprs_sub, tprs_sub = [1.0], [1.0]
        for thr in sorted(thresholds, reverse=True):
            predicted_pos = scores >= thr
            tp = np.logical_and(predicted_pos, binary_true).sum()
            fp = np.logical_and(predicted_pos, ~binary_true.astype(bool)).sum()
            fn = np.logical_and(~predicted_pos, binary_true.astype(bool)).sum()
            tn = np.logical_and(~predicted_pos, ~binary_true.astype(bool)).sum()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fprs_sub.append(fpr)
            tprs_sub.append(tpr)
        fprs_sub.append(0.0)
        tprs_sub.append(0.0)

        # interpolate onto common grid
        interp_tpr = np.interp(all_fpr, fprs_sub[::-1], tprs_sub[::-1])
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0
    # auc = float(np.trapz(mean_tpr, all_fpr))
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

    train_projections = pca.train_projections   # already stored in the model

    # --- predict each test sample ---
    y_pred = []
    distances = []
    for vec in X_test:
        proj = pca.transform(vec.reshape(1, -1))              # (1, d)
        dists = np.linalg.norm(train_projections - proj, axis=1)
        idx = int(np.argmin(dists))
        y_pred.append(int(y_train[idx]))
        distances.append(float(dists[idx]))

    y_pred = np.array(y_pred)
    distances = np.array(distances)

    metrics = compute_metrics(y_test, y_pred)

    fpr, tpr, auc = _compute_roc_macro(
        y_test, distances, y_train, train_projections, pca, X_test
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