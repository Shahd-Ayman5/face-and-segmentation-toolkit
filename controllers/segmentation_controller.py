"""
controllers/segmentation_controller.py
----------------------------------------
Controls the Segmentation tab in main.ui.

Responsibilities
----------------
- Load a colour or grayscale image via a file dialog.
- Let the user click on the input canvas to place seed points.
- Run Region Growing off the GUI thread (same QThread + mp.Process
  pattern used throughout the project).
- Display input (with seed markers drawn on it) and output images.
- Save the result to disk.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QBrush, QColor
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QListWidgetItem

from core.segmentation.region_growing import region_growing


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(image: np.ndarray,
               seeds: list[tuple[int, int]],
               threshold: int,
               queue: mp.Queue) -> None:
    try:
        result = region_growing(image, seeds, threshold)
        queue.put(result)
    except Exception as exc:
        queue.put(None)


class _SegWorker(QThread):
    finished = pyqtSignal(object)

    def __init__(self, image, seeds, threshold):
        super().__init__()
        self._image     = image
        self._seeds     = seeds
        self._threshold = threshold

    def run(self):
        q = mp.Queue()
        p = mp.Process(target=_worker_fn,
                       args=(self._image, self._seeds, self._threshold, q))
        p.start()
        while True:
            if not q.empty():
                self.finished.emit(q.get())
                break
            self.msleep(40)
        p.join()


# ─────────────────────────────────────────────────────────────────────────────
# Clickable canvas helper
# ─────────────────────────────────────────────────────────────────────────────

class _ClickableLabel:
    """
    Monkey-patches a QLabel to emit (x, y) in *image* coordinates on click.
    We do it without subclassing so we don't need a custom widget in the .ui.
    """
    def __init__(self, label, callback):
        self._label    = label
        self._callback = callback
        self._img_w    = 1
        self._img_h    = 1
        label.mousePressEvent = self._on_click

    def set_image_size(self, w: int, h: int):
        self._img_w = w
        self._img_h = h

    def _on_click(self, event):
        lw = self._label.width()
        lh = self._label.height()
        if lw == 0 or lh == 0:
            return

        # The pixmap is scaled keeping aspect ratio and centred
        scale = min(lw / self._img_w, lh / self._img_h)
        disp_w = int(self._img_w * scale)
        disp_h = int(self._img_h * scale)
        off_x  = (lw - disp_w) // 2
        off_y  = (lh - disp_h) // 2

        px = event.x() - off_x
        py = event.y() - off_y
        if 0 <= px < disp_w and 0 <= py < disp_h:
            img_x = int(px / scale)
            img_y = int(py / scale)
            self._callback(img_x, img_y)


# ─────────────────────────────────────────────────────────────────────────────
# Controller
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationController(QObject):
    status_message = pyqtSignal(str)

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window : QMainWindow      = window
        self._image  : np.ndarray | None = None
        self._result : np.ndarray | None = None
        self._seeds  : list[tuple[int, int]] = []
        self._worker : _SegWorker | None = None

        # Wrap input canvas for click events
        self._canvas = _ClickableLabel(
            window.segInputCanvas, self._on_canvas_click
        )

    # ------------------------------------------------------------------
    def bind_ui(self):
        w = self._window
        w.segBtnLoad.clicked.connect(self._load_image)
        w.segBtnApply.clicked.connect(self._apply)
        w.segBtnSave.clicked.connect(self._save)
        w.segBtnClearSeeds.clicked.connect(self._clear_seeds)

    # ------------------------------------------------------------------
    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self._window, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            self._set_status("❌ Could not read image.", error=True)
            return

        # Ensure at least 3 channels
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self._image  = img
        self._result = None
        self._seeds  = []
        self._window.segSeedsList.clear()

        h, w = img.shape[:2]
        self._canvas.set_image_size(w, h)

        self._refresh_input_canvas()
        self._window.segOutputCanvas.setText("Result will appear here")
        self._window.segOutputCanvas.setPixmap(QPixmap())

        name = Path(path).name
        self._window.segImageNameLbl.setText(name)
        self._window.segBtnApply.setEnabled(False)   # need seeds first
        self._window.segBtnSave.setEnabled(False)
        self._set_status(f"Loaded: {name}  |  Click image to add seed points.")

    # ------------------------------------------------------------------
    def _on_canvas_click(self, x: int, y: int):
        if self._image is None:
            return
        self._seeds.append((x, y))

        # Update list widget
        item = QListWidgetItem(f"Seed {len(self._seeds)}: ({x}, {y})")
        self._window.segSeedsList.addItem(item)

        self._refresh_input_canvas()
        self._window.segBtnApply.setEnabled(True)
        self._set_status(
            f"{len(self._seeds)} seed(s) placed — ready to apply."
        )

    # ------------------------------------------------------------------
    def _clear_seeds(self):
        self._seeds = []
        self._window.segSeedsList.clear()
        self._window.segBtnApply.setEnabled(False)
        self._refresh_input_canvas()
        self._set_status("Seeds cleared.")

    # ------------------------------------------------------------------
    def _apply(self):
        if self._image is None or not self._seeds:
            return

        threshold = self._window.regionThreshSpin.value()

        self._window.segBtnApply.setEnabled(False)
        self._window.segBtnLoad.setEnabled(False)
        self._set_status("⏳ Growing regions…")

        self._worker = _SegWorker(self._image, self._seeds, threshold)
        self._worker.finished.connect(self._on_result)
        self._worker.start()

    # ------------------------------------------------------------------
    def _on_result(self, result):
        self._window.segBtnApply.setEnabled(True)
        self._window.segBtnLoad.setEnabled(True)

        if result is None:
            self._set_status("❌ Segmentation failed.", error=True)
            return

        self._result = result
        _show_bgr_on_canvas(result, self._window.segOutputCanvas)
        self._window.segBtnSave.setEnabled(True)
        self._set_status(
            f"✅ Done — {len(self._seeds)} region(s) grown, "
            f"threshold = {self._window.regionThreshSpin.value()}"
        )

    # ------------------------------------------------------------------
    def _save(self):
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save Result", "segmented.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self._result)
            self._set_status(f"💾 Saved to {Path(path).name}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_input_canvas(self):
        """Display current image with seed markers drawn on it."""
        if self._image is None:
            return
        vis = self._image.copy()

        # Draw each seed as a cross + circle
        for idx, (sx, sy) in enumerate(self._seeds):
            colour_map = [
                (0,   0,   255),
                (0,   255, 0  ),
                (255, 0,   0  ),
                (0,   200, 255),
                (255, 0,   200),
                (255, 200, 0  ),
            ]
            bgr = colour_map[idx % len(colour_map)]
            r   = 8
            cv2.circle(vis, (sx, sy), r, bgr, 2)
            cv2.line(vis, (sx - r, sy), (sx + r, sy), bgr, 2)
            cv2.line(vis, (sx, sy - r), (sx, sy + r), bgr, 2)
            cv2.putText(vis, str(idx + 1), (sx + r + 2, sy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1,
                        cv2.LINE_AA)

        _show_bgr_on_canvas(vis, self._window.segInputCanvas)

    def _set_status(self, msg: str, error: bool = False):
        colour = "#cc4444" if error else "#88cc88"
        self._window.segStatusLbl.setStyleSheet(
            f"color: {colour}; font-size: 10pt;"
        )
        self._window.segStatusLbl.setText(msg)
        self.status_message.emit(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Shared canvas utility
# ─────────────────────────────────────────────────────────────────────────────

def _show_bgr_on_canvas(img: np.ndarray, label) -> None:
    """Convert a BGR/BGRA numpy array to QPixmap and fit it into `label`."""
    lw = max(label.width(),  1)
    lh = max(label.height(), 1)

    h, w = img.shape[:2]
    c    = img.shape[2] if img.ndim == 3 else 1

    if c == 4:
        # BGRA → RGBA
        rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
    else:
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)

    pix = QPixmap.fromImage(qimg).scaled(
        lw, lh,
        aspectRatioMode=Qt.KeepAspectRatio,
        transformMode=Qt.SmoothTransformation
    )
    label.setPixmap(pix)