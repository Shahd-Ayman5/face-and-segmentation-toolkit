from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QProgressBar, QWidget, QSizePolicy, QMessageBox,
    QScrollArea
)
from PyQt5.QtGui import QPixmap

from core.face.roc import evaluate_model, EvaluationResult
from core.face.visualization import plot_roc, plot_metrics_bar, plot_confusion_matrix


# ──────────────────────────────────────────────────────────────────────────────
# Background worker
# ──────────────────────────────────────────────────────────────────────────────

class _EvalWorker(QThread):
    finished  = pyqtSignal(object)   # EvaluationResult
    error     = pyqtSignal(str)

    def __init__(self, model_path: str):
        super().__init__()
        self._model_path = model_path

    def run(self):
        try:
            result = evaluate_model(self._model_path)
            self.finished.emit(result)
        except Exception as exc:
            traceback.print_exc()
            self.error.emit(traceback.format_exc())


# ──────────────────────────────────────────────────────────────────────────────
# Dialog
# ──────────────────────────────────────────────────────────────────────────────

class EvaluationDialog(QDialog):

    _STYLESHEET = """
        QDialog        { background:#1e1e1e; color:#e0e0e0; }
        QTabWidget::pane { border:0; }
        QTabBar::tab   { background:#000; color:white; padding:8px 18px;
                         border:1px solid #222; font-size:10pt; }
        QTabBar::tab:selected { background:#1e1e1e; border-bottom:2px solid #7c6af7; }
        QTabBar::tab:hover    { background:#2a2a2a; }
        QLabel         { color:#e0e0e0; }
        QPushButton    { background:#2a2a2a; color:#e0e0e0;
                         border:1px solid #444; padding:6px 16px; border-radius:4px; }
        QPushButton:hover { background:#3a3a3a; }
        QProgressBar   { background:#2a2a2a; border:1px solid #444;
                         border-radius:4px; text-align:center; color:#e0e0e0; }
        QProgressBar::chunk { background:#7c6af7; border-radius:4px; }
        QScrollArea    { border:none; }
    """

    def __init__(self, model_path: str, parent=None):
        super().__init__(parent)
        self._model_path = model_path
        self._result: Optional[EvaluationResult] = None

        self.setWindowTitle("Face Recognition – Evaluation")
        self.setMinimumSize(750, 600)
        self.resize(900, 700)
        self.setStyleSheet(self._STYLESHEET)

        self._build_ui()
        self._start_evaluation()

    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # loading area
        self._loading_widget = QWidget()
        ll = QVBoxLayout(self._loading_widget)
        self._status_lbl = QLabel("Running evaluation on test set…\n(this may take 10–30 seconds)")
        self._status_lbl.setAlignment(Qt.AlignCenter)
        self._status_lbl.setWordWrap(True)
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)
        ll.addStretch()
        ll.addWidget(self._status_lbl)
        ll.addSpacing(10)
        ll.addWidget(self._progress)
        ll.addStretch()
        root.addWidget(self._loading_widget)

        # results area
        self._results_widget = QWidget()
        self._results_widget.setVisible(False)
        rl = QVBoxLayout(self._results_widget)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        self._summary_lbl = QLabel()
        self._summary_lbl.setAlignment(Qt.AlignCenter)
        self._summary_lbl.setStyleSheet("font-size:11pt; font-weight:bold; padding:4px;")
        self._summary_lbl.setWordWrap(True)
        rl.addWidget(self._summary_lbl)

        self._tabs = QTabWidget()
        rl.addWidget(self._tabs)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        rl.addLayout(btn_row)

        root.addWidget(self._results_widget)

    def _make_image_tab(self, pixmap: QPixmap) -> QWidget:
        outer = QWidget()
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QLabel()
        inner.setAlignment(Qt.AlignCenter)
        inner.setPixmap(pixmap)
        scroll.setWidget(inner)
        lay.addWidget(scroll)
        return outer

    # ------------------------------------------------------------------
    def _start_evaluation(self):
        self._worker = _EvalWorker(self._model_path)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_finished(self, result: EvaluationResult):
        self._result = result
        m = result.metrics

        self._summary_lbl.setText(
            f"Accuracy: {m.accuracy:.1%}   |   "
            f"Precision: {m.precision:.3f}   |   "
            f"Recall: {m.recall:.3f}   |   "
            f"F1: {m.f1:.3f}   |   "
            f"AUC: {result.auc:.3f}"
        )

        roc_pix = plot_roc(result.fpr, result.tpr, result.auc)
        bar_pix = plot_metrics_bar(m.accuracy, m.precision, m.recall, m.f1)
        cm_pix  = plot_confusion_matrix(m.confusion_matrix, m.labels)

        self._tabs.addTab(self._make_image_tab(roc_pix), "ROC Curve")
        self._tabs.addTab(self._make_image_tab(bar_pix), "Metrics")
        self._tabs.addTab(self._make_image_tab(cm_pix),  "Confusion Matrix")

        self._loading_widget.setVisible(False)
        self._results_widget.setVisible(True)

    def _on_error(self, full_traceback: str):
        self._progress.setVisible(False)
        last_line = full_traceback.strip().splitlines()[-1]
        self._status_lbl.setText(
            f"Evaluation failed:\n{last_line}\n\nSee console for full traceback."
        )
        self._status_lbl.setStyleSheet("color:#f77c6a; font-size:9pt;")

        QMessageBox.critical(
            self,
            "Evaluation Error",
            f"The evaluation failed:\n\n{last_line}\n\n"
            f"Common causes:\n"
            f"  • pca_model.pkl not found at expected path\n"
            f"  • ORL dataset folder not found\n"
            f"  • matplotlib not installed  (pip install matplotlib)"
        )