"""
controllers/thresholding_controller.py
---------------------------------------
Controls the Thresholding tab in main.ui.

Responsibilities
----------------
- Load a grayscale image via a file dialog.
- Let the user pick a thresholding method from the combo-box
  (Optimal Thresholding, Spectral Thresholding, Otsu Thresholding).
- Run the algorithm off the GUI thread (QThread + subprocess via
  multiprocessing.Queue – same pattern as the rest of the project).
- Display input image, output image, and a histogram with threshold
  lines drawn on it.
- Save the result to disk.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt5.QtWidgets import QFileDialog, QMainWindow

# Algorithms
from core.Thresholding.spectral_thresholding import spectral_threshold
from core.Thresholding.otsu import otsu_threshold
from core.Thresholding.optimal import apply_optimal_threshold
from core.Thresholding.local_thresholding import local_threshold

# ─────────────────────────────────────────────────────────────────────────────
# Worker – runs in a QThread, algorithm in a child Process
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(gray: np.ndarray, method: str, n_modes: int, queue: mp.Queue, 
               block_size: int = 11, C: int = 2):
    try:
        method_l = method.lower()

        if "otsu" in method_l:
            threshold_value, result = otsu_threshold(gray)
            queue.put(("otsu", {"result": result, "thresholds": [threshold_value]}))

        elif "optimal" in method_l:
            result_dict = apply_optimal_threshold(gray)
            queue.put(("optimal", result_dict))

        elif "spectral" in method_l:
            result, thresholds = spectral_threshold(gray, n_modes=n_modes)
            queue.put(("spectral", {"result": result, "thresholds": thresholds}))

        elif "local" in method_l:
            result = local_threshold(gray, block_size=block_size, C=C)
            queue.put(("local", {"result": result, "thresholds": []}))

        else:
            raise ValueError(f"Unknown method: {method}")

    except Exception as exc:
        queue.put(("error", str(exc)))


class _ThresholdWorker(QThread):
    finished = pyqtSignal(str, object)   # (kind, payload)

    def __init__(self, gray: np.ndarray, method: str, n_modes: int, 
                 block_size: int = 11, C: int = 2):
        super().__init__()
        self._gray       = gray
        self._method     = method
        self._n_modes    = n_modes
        self._block_size = block_size
        self._C          = C

    def run(self):
        q = mp.Queue()
        p = mp.Process(target=_worker_fn,
                       args=(self._gray, self._method, self._n_modes, q,
                             self._block_size, self._C))
        p.start()
        while True:
            if not q.empty():
                kind, payload = q.get()
                self.finished.emit(kind, payload)
                break
            self.msleep(40)
        p.join()


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class ThresholdingController(QObject):
    status_message = pyqtSignal(str)

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window      = window
        self._gray_image  : np.ndarray | None = None
        self._result      : np.ndarray | None = None
        self._thresholds  : list[int]          = []
        self._worker      : _ThresholdWorker  | None = None

    # ------------------------------------------------------------------
    def bind_ui(self):
        w = self._window
        w.threshBtnLoad.clicked.connect(self._load_image)
        w.threshBtnApply.clicked.connect(self._apply)
        w.threshBtnSave.clicked.connect(self._save)
        w.threshMethodCombo.currentTextChanged.connect(self._on_method_changed)
        self._on_method_changed(w.threshMethodCombo.currentText())

    # ------------------------------------------------------------------
    def _on_method_changed(self, method: str):
        method = method.lower()

        self._window.optimalParamsGroup.setVisible("optimal" in method)
        self._window.spectralParamsGroup.setVisible("spectral" in method)
        self._window.localParamsGroup.setVisible("local" in method)

        # Otsu has no tunable parameters in this UI.
        if "otsu" in method:
            self._window.optimalParamsGroup.setVisible(False)
            self._window.spectralParamsGroup.setVisible(False)
            self._window.localParamsGroup.setVisible(False)

    # ------------------------------------------------------------------
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

        self._gray_image = img
        self._result     = None
        self._thresholds = []

        # Display input
        self._show_on_canvas(img, self._window.threshInputCanvas, gray=True)

        # Clear output
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

    # ------------------------------------------------------------------
    def _apply(self):
        if self._gray_image is None:
            return

        method = self._window.threshMethodCombo.currentText()
        n_modes = self._window.spectralModesSpin.value()
        block_size = self._window.localBlockSizeSpin.value()
        C = self._window.localConstantSpin.value()

        # Disable controls while running
        self._window.threshBtnApply.setEnabled(False)
        self._window.threshBtnLoad.setEnabled(False)
        self._set_status("⏳ Processing…")

        self._worker = _ThresholdWorker(self._gray_image, method, n_modes,
                                       block_size, C)
        self._worker.finished.connect(self._on_result)
        self._worker.start()

    # ------------------------------------------------------------------
    def _on_result(self, kind: str, payload):
        self._window.threshBtnApply.setEnabled(True)
        self._window.threshBtnLoad.setEnabled(True)

        if kind == "error":
            self._set_status(f"❌ Error: {payload}", error=True)
            return

        if kind == "otsu":
            result     = payload["result"]
            thresholds = payload["thresholds"]

            self._result     = result
            self._thresholds = thresholds
            self._show_on_canvas(result, self._window.threshOutputCanvas, gray=True)
            self._draw_histogram(self._gray_image, thresholds,
                                 self._window.threshHistCanvas)
            self._window.threshResultInfoLbl.setText(
                f"Threshold: {thresholds[0]}"
            )
            self._set_status(f"✅ Otsu done — T={thresholds[0]}")

        elif kind == "optimal":
            binary     = payload["binary"]
            threshold  = payload["threshold"]
            iterations = payload["iterations"]
            history    = payload["history"]

            self._result     = binary
            self._thresholds = [threshold]
            self._show_on_canvas(binary, self._window.threshOutputCanvas, gray=True)
            self._draw_histogram(self._gray_image, [threshold],
                                 self._window.threshHistCanvas)
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
            result     = payload["result"]
            thresholds = payload["thresholds"]

            self._result     = result
            self._thresholds = thresholds
            self._show_on_canvas(result, self._window.threshOutputCanvas, gray=True)
            self._draw_histogram(self._gray_image, thresholds,
                                 self._window.threshHistCanvas)
            n_regions = len(thresholds) + 1
            self._window.threshResultInfoLbl.setText(
                f"{n_regions} regions  |  Thresholds: {thresholds}"
            )
            self._set_status(
                f"✅ Spectral done — {n_regions} regions, "
                f"thresholds: {thresholds}"
            )

        elif kind == "local":
            result = payload["result"]

            self._result     = result
            self._thresholds = []
            self._show_on_canvas(result, self._window.threshOutputCanvas, gray=True)
            self._draw_histogram(self._gray_image, [],
                                 self._window.threshHistCanvas)
            block_size = self._window.localBlockSizeSpin.value()
            C = self._window.localConstantSpin.value()
            self._window.threshResultInfoLbl.setText(
                f"Block Size: {block_size}  |  Constant (C): {C}"
            )
            self._set_status(
                f"✅ Local (Adaptive) done — "
                f"Block Size={block_size}, C={C}"
            )

        self._window.threshBtnSave.setEnabled(True)

    # ------------------------------------------------------------------
    def _save(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save Result", "thresholded.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self._result)
            self._set_status(f"💾 Saved to {Path(path).name}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, msg: str, error: bool = False):
        colour = "#cc4444" if error else "#88cc88"
        self._window.threshStatusLbl.setStyleSheet(
            f"color: {colour}; font-size: 10pt;"
        )
        self._window.threshStatusLbl.setText(msg)
        self.status_message.emit(msg)

    # ------------------------------------------------------------------
    @staticmethod
    def _show_on_canvas(img: np.ndarray, label, *, gray: bool = False):
        """Fit `img` into `label` preserving aspect ratio."""
        lw = max(label.width(),  1)
        lh = max(label.height(), 1)

        if gray:
            h, w     = img.shape[:2]
            qimg     = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, c  = img.shape
            bytes_per_line = c * w
            fmt = QImage.Format_RGB888 if c == 3 else QImage.Format_RGBA8888
            # OpenCV is BGR; swap channels for Qt
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, bytes_per_line, fmt)

        pix = QPixmap.fromImage(qimg).scaled(
            lw, lh,
            aspectRatioMode=Qt.KeepAspectRatio,
            transformMode=Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_histogram(gray: np.ndarray,
                        thresholds: list[int],
                        label) -> None:
        """
        Draw the image histogram and overlay vertical lines at each threshold.
        Renders into a QPixmap and sets it on `label`.
        """
        W = max(label.width(),  512)
        H = max(label.height(), 128)

        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        hist_max = hist.max() if hist.max() > 0 else 1

        pix = QPixmap(W, H)
        pix.fill(QColor("#0a0a0a"))

        painter = QPainter(pix)

        # ── Draw histogram bars ──────────────────────────────────────
        bar_w = max(1, W // 256)
        bar_colour = QColor("#4488cc")
        painter.setPen(Qt.NoPen)
        painter.setBrush(bar_colour)

        for i, count in enumerate(hist):
            bar_h = int((count / hist_max) * (H - 10))
            x     = int(i * W / 256)
            painter.drawRect(x, H - bar_h, bar_w, bar_h)

        # ── Draw threshold lines ─────────────────────────────────────
        colours = ["#ff4444", "#ffaa00", "#44ff44",
                   "#ff44ff", "#44ffff", "#ffff44"]
        font = QFont("Arial", 8)
        painter.setFont(font)

        for idx, t in enumerate(thresholds):
            x   = int(t * W / 256)
            col = QColor(colours[idx % len(colours)])
            pen = QPen(col, 2)
            painter.setPen(pen)
            painter.drawLine(x, 0, x, H)

            # Label
            painter.drawText(x + 2, 14 + idx * 14, f"T{idx+1}={t}")

        painter.end()
        label.setPixmap(pix)
