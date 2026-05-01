"""
controllers/thresholding_tab_controller.py
-------------------------------------------
Unified controller for the Thresholding tab.

Manages the combo-box switch between:
  • Optimal Thresholding   (index 0) — Lolo's algorithm
  • Spectral Thresholding  (index 1) — Shahod's algorithm

All shared widgets (Load, Apply, Save, canvases, histogram, status)
live once in the .ui; this controller owns them and delegates the
actual algorithm work to the two sub-controllers.

Sub-controllers no longer bind_ui() themselves — this class is the
single point of contact with the window.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow

# ── algorithms ────────────────────────────────────────────────────────────────
from core.thresholding.optimal import apply_optimal_threshold
from core.Thresholding.spectral_thresholding import spectral_threshold

METHOD_OPTIMAL  = 0
METHOD_SPECTRAL = 1


# ─────────────────────────────────────────────────────────────────────────────
# Workers
# ─────────────────────────────────────────────────────────────────────────────

def _optimal_worker_fn(gray: np.ndarray, queue: mp.Queue) -> None:
    try:
        queue.put(("optimal", apply_optimal_threshold(gray)))
    except Exception as exc:
        queue.put(("error", str(exc)))


def _spectral_worker_fn(gray: np.ndarray, n_modes: int, queue: mp.Queue) -> None:
    try:
        result, thresholds = spectral_threshold(gray, n_modes=n_modes)
        queue.put(("spectral", {"binary": result, "thresholds": thresholds}))
    except Exception as exc:
        queue.put(("error", str(exc)))


class _ThreshWorker(QThread):
    finished = pyqtSignal(str, object)   # (kind, payload)

    def __init__(self, target_fn, args: tuple):
        super().__init__()
        self._target = target_fn
        self._args   = args

    def run(self):
        q = mp.Queue()
        p = mp.Process(target=self._target, args=(*self._args, q))
        p.start()
        while True:
            if not q.empty():
                kind, payload = q.get()
                self.finished.emit(kind, payload)
                break
            self.msleep(40)
        p.join()


# ─────────────────────────────────────────────────────────────────────────────
# Unified Controller
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdingTabController(QObject):
    status_message = pyqtSignal(str)

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window     = window
        self._gray       : np.ndarray | None = None
        self._result_img : np.ndarray | None = None
        self._worker     : _ThreshWorker | None = None

    # ------------------------------------------------------------------ bind
    def bind_ui(self):
        w = self._window
        w.threshBtnLoad.clicked.connect(self._load_image)
        w.threshBtnApply.clicked.connect(self._apply)
        w.threshBtnSave.clicked.connect(self._save)
        w.threshMethodCombo.currentIndexChanged.connect(self._on_method_changed)

        # Show the correct params group on startup
        self._on_method_changed(w.threshMethodCombo.currentIndex())

    # --------------------------------------------------------- method switch
    def _on_method_changed(self, index: int):
        w = self._window
        is_optimal  = (index == METHOD_OPTIMAL)
        is_spectral = (index == METHOD_SPECTRAL)

        w.optimalParamsGroup.setVisible(is_optimal)
        w.spectralParamsGroup.setVisible(is_spectral)

        # Reset outputs when switching so stale results are not shown
        self._result_img = None
        w.threshOutputCanvas.setText("Result will appear here")
        w.threshOutputCanvas.setPixmap(QPixmap())
        w.threshHistCanvas.setText(
            "Histogram will appear here after applying"
        )
        w.threshHistCanvas.setPixmap(QPixmap())
        w.threshResultInfoLbl.setText("")
        w.threshBtnSave.setEnabled(False)

        if self._gray is not None:
            w.threshBtnApply.setEnabled(True)

    # ------------------------------------------------------------------ load
    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Open Grayscale Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            self._set_status("❌ Could not read image.", error=True)
            return

        self._gray       = img
        self._result_img = None

        _show_gray(img, self._window.threshInputCanvas)

        self._window.threshOutputCanvas.setText("Result will appear here")
        self._window.threshOutputCanvas.setPixmap(QPixmap())
        self._window.threshHistCanvas.setText(
            "Histogram with threshold lines will appear here after applying"
        )
        self._window.threshHistCanvas.setPixmap(QPixmap())
        self._window.threshResultInfoLbl.setText("")

        name = Path(path).name
        self._window.threshImageNameLbl.setText(name)
        self._window.threshBtnApply.setEnabled(True)
        self._window.threshBtnSave.setEnabled(False)
        self._set_status(f"Loaded: {name}")

    # ----------------------------------------------------------------- apply
    def _apply(self):
        if self._gray is None:
            return

        self._window.threshBtnApply.setEnabled(False)
        self._window.threshBtnLoad.setEnabled(False)
        self._set_status("⏳ Processing…")

        method = self._window.threshMethodCombo.currentIndex()

        if method == METHOD_OPTIMAL:
            self._worker = _ThreshWorker(
                _optimal_worker_fn,
                (self._gray,)
            )
        else:
            n_modes = self._window.spectralModesSpin.value()
            self._worker = _ThreshWorker(
                _spectral_worker_fn,
                (self._gray, n_modes)
            )

        self._worker.finished.connect(self._on_result)
        self._worker.start()

    # --------------------------------------------------------------- result
    def _on_result(self, kind: str, payload):
        self._window.threshBtnApply.setEnabled(True)
        self._window.threshBtnLoad.setEnabled(True)

        if kind == "error":
            self._set_status(f"❌ Error: {payload}", error=True)
            return

        if kind == "optimal":
            binary     = payload["binary"]
            threshold  = payload["threshold"]
            iterations = payload["iterations"]
            history    = payload["history"]

            self._result_img = binary
            _show_gray(binary, self._window.threshOutputCanvas)
            _draw_histogram(self._gray, [threshold],
                            self._window.threshHistCanvas, ["#ff4444"])

            self._window.threshResultInfoLbl.setText(
                f"Threshold: {threshold}  |  "
                f"Iterations: {iterations}  |  "
                "History: " + " → ".join(f"{v:.1f}" for v in history)
            )
            self._set_status(
                f"✅ Optimal done — T={threshold}, "
                f"converged in {iterations} iteration(s)"
            )

        elif kind == "spectral":
            binary     = payload["binary"]
            thresholds = payload["thresholds"]

            self._result_img = binary
            _show_gray(binary, self._window.threshOutputCanvas)
            _draw_histogram(self._gray, thresholds,
                            self._window.threshHistCanvas)

            n_regions = len(thresholds) + 1
            self._window.threshResultInfoLbl.setText(
                f"{n_regions} regions  |  Thresholds: {thresholds}"
            )
            self._set_status(
                f"✅ Spectral done — {n_regions} regions, "
                f"thresholds: {thresholds}"
            )

        self._window.threshBtnSave.setEnabled(True)

    # ------------------------------------------------------------------ save
    def _save(self):
        if self._result_img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save Result", "threshold_result.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self._result_img)
            self._set_status(f"💾 Saved to {Path(path).name}")

    # ---------------------------------------------------------------- helpers
    def _set_status(self, msg: str, error: bool = False):
        colour = "#cc4444" if error else "#88cc88"
        self._window.threshStatusLbl.setStyleSheet(
            f"color:{colour}; font-size:10pt;"
        )
        self._window.threshStatusLbl.setText(msg)
        self.status_message.emit(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Canvas helpers (private to this module)
# ─────────────────────────────────────────────────────────────────────────────

def _show_gray(img: np.ndarray, label) -> None:
    lw = max(label.width(), 1)
    lh = max(label.height(), 1)
    h, w = img.shape[:2]
    qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    pix  = QPixmap.fromImage(qimg).scaled(
        lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )
    label.setPixmap(pix)


def _draw_histogram(
    gray: np.ndarray,
    thresholds: list[int],
    label,
    colours: list[str] | None = None,
) -> None:
    W = max(label.width(),  512)
    H = max(label.height(), 128)

    hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    hist_max = hist.max() if hist.max() > 0 else 1

    if colours is None:
        colours = ["#ff4444", "#ffaa00", "#44ff44",
                   "#ff44ff", "#44ffff", "#ffff44"]

    pix = QPixmap(W, H)
    pix.fill(QColor("#0a0a0a"))
    painter = QPainter(pix)

    bar_w = max(1, W // 256)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor("#4488cc"))
    for i, count in enumerate(hist):
        bar_h = int((count / hist_max) * (H - 10))
        painter.drawRect(int(i * W / 256), H - bar_h, bar_w, bar_h)

    painter.setFont(QFont("Arial", 8))
    for idx, t in enumerate(thresholds):
        x = int(t * W / 256)
        painter.setPen(QPen(QColor(colours[idx % len(colours)]), 2))
        painter.drawLine(x, 0, x, H)
        painter.drawText(x + 2, 14 + idx * 14, f"T={t}")

    painter.end()
    label.setPixmap(pix)