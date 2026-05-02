from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed – we embed into Qt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from io import BytesIO

from PyQt5.QtGui import QPixmap, QImage


# ──────────────────────────────────────────────────────────────────────────────
# Internal helper: figure → QPixmap
# ──────────────────────────────────────────────────────────────────────────────

def _fig_to_pixmap(fig: plt.Figure) -> QPixmap:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    data = buf.read()
    qimg = QImage.fromData(data)
    return QPixmap.fromImage(qimg)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#1e1e1e"
DARK_AX   = "#2a2a2a"
ACCENT    = "#7c6af7"
ACCENT2   = "#f77c6a"
TEXT      = "#e0e0e0"
GRID      = "#3a3a3a"


def _apply_dark_style(fig: plt.Figure, axes):
    """Apply a dark theme consistent with the app's QSS."""
    fig.patch.set_facecolor(DARK_BG)
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(DARK_AX)
        ax.tick_params(colors=TEXT)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linestyle="--", linewidth=0.6)


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc: float) -> QPixmap:
    """Return a QPixmap with the ROC curve."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (macro one-vs-rest)")
    ax.legend(facecolor=DARK_AX, edgecolor=GRID, labelcolor=TEXT)
    _apply_dark_style(fig, ax)
    pixmap = _fig_to_pixmap(fig)
    plt.close(fig)
    return pixmap


def plot_metrics_bar(accuracy: float, precision: float,
                     recall: float, f1: float) -> QPixmap:
    """Return a QPixmap with a simple bar chart of the four scalar metrics."""
    names  = ["Accuracy", "Precision", "Recall", "F1"]
    values = [accuracy, precision, recall, f1]
    colors = [ACCENT, ACCENT2, "#6af77c", "#f7e26a"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(names, values, color=colors, width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Classification Metrics (macro-averaged)")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9)

    _apply_dark_style(fig, ax)
    pixmap = _fig_to_pixmap(fig)
    plt.close(fig)
    return pixmap


def plot_confusion_matrix(cm: np.ndarray, labels: np.ndarray,
                          max_labels: int = 20) -> QPixmap:
    """
    Return a QPixmap with a heatmap of the confusion matrix.
    If there are >max_labels classes we trim the display to keep it readable.
    """
    if len(labels) > max_labels:
        cm     = cm[:max_labels, :max_labels]
        labels = labels[:max_labels]
        title  = f"Confusion Matrix (first {max_labels} subjects)"
    else:
        title  = "Confusion Matrix"

    n   = len(labels)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.4), max(4, n * 0.35)))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"s{l}" for l in labels], rotation=90, fontsize=7)
    ax.set_yticklabels([f"s{l}" for l in labels], fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # annotate cells only when small enough
    if n <= 15:
        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "#333",
                        fontsize=7)

    _apply_dark_style(fig, ax)
    fig.tight_layout()
    pixmap = _fig_to_pixmap(fig)
    plt.close(fig)
    return pixmap