"""
controllers/segmentation_controller.py
----------------------------------------
Controls the Segmentation tab in main.ui.

Responsibilities
----------------
- Load a colour or grayscale image via a file dialog.
- Let the user click on the input canvas to place seed points.
- Run segmentation off the GUI thread (same QThread + mp.Process
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
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QListWidgetItem

from core.segmentation.kmeans import apply_kmeans_segmentation
from core.segmentation.region_growing import region_growing
from core.segmentation.mean_shift import mean_shift_segmentation
from core.segmentation.agglomerative import agglomerative_segmentation


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _worker_fn(image: np.ndarray,
               method: str,
               seeds: list[tuple[int, int]],
               threshold: int,
               km_clusters: int,
               ms_spatial_radius: int,
               ms_color_radius: int,
               ms_max_iter: int,
               aggl_clusters: int,
               aggl_block_size: int,
               aggl_alpha: float,
               queue: mp.Queue) -> None:
    try:
        method = method.lower()

        if "region" in method:
            result = region_growing(image, seeds, threshold)
            queue.put(("result", result, None))
        elif "mean" in method and "shift" in method:
            original_h, original_w = image.shape[:2]
            small = cv2.resize(image, (100, 100), interpolation=cv2.INTER_AREA)

            result_small = mean_shift_segmentation(
                small,
                spatial_radius=ms_spatial_radius,
                color_radius=ms_color_radius,
                max_iter=ms_max_iter,
                progress_queue=queue,
            )
            result = cv2.resize(
                result_small,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )
            queue.put(("result", result, None))
        elif "k" in method and "means" in method:
            seg_result = apply_kmeans_segmentation(
                image, k=km_clusters, random_state=42
            )
            queue.put(("kmeans", seg_result, None))
        elif "agglomerative" in method:
            original_h, original_w = image.shape[:2]

            small = image
            result_small = agglomerative_segmentation(
                small,
                n_clusters=aggl_clusters,
                block_size=aggl_block_size,
                alpha=aggl_alpha,
                progress_queue=queue,
            )

            result = cv2.resize(
                result_small,
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )

            queue.put(("result", result, None))
        else:
            raise ValueError(f"Unknown segmentation method: {method}")

    except Exception as exc:
        queue.put(("error", None, str(exc)))


class _SegWorker(QThread):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self,
                 image,
                 method,
                 seeds,
                 threshold,
                 km_clusters,
                 ms_spatial_radius,
                 ms_color_radius,
                 ms_max_iter,
                 aggl_clusters=8,
                 aggl_block_size=3,
                 aggl_alpha=0.5):
        super().__init__()
        self._image             = image
        self._method            = method
        self._seeds             = seeds
        self._threshold         = threshold
        self._km_clusters       = km_clusters
        self._ms_spatial_radius = ms_spatial_radius
        self._ms_color_radius   = ms_color_radius
        self._ms_max_iter       = ms_max_iter
        self._aggl_clusters     = aggl_clusters
        self._aggl_block_size   = aggl_block_size
        self._aggl_alpha        = aggl_alpha

    def run(self):
        q = mp.Queue()
        p = mp.Process(target=_worker_fn,
                       args=(
                           self._image,
                           self._method,
                           self._seeds,
                           self._threshold,
                           self._km_clusters,
                           self._ms_spatial_radius,
                           self._ms_color_radius,
                           self._ms_max_iter,
                           self._aggl_clusters,
                           self._aggl_block_size,
                           self._aggl_alpha,
                           q,
                       ))
        p.start()
        while True:
            if not q.empty():
                message = q.get()
                if isinstance(message, tuple) and message and message[0] == "progress":
                    method_name = self._method
                    self.progress.emit(f"⏳ {method_name} processing... {message[1]}%")
                    continue
                self.finished.emit(message)
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
        w.segMethodCombo.currentTextChanged.connect(self._on_method_changed)
        self._on_method_changed(w.segMethodCombo.currentText())

    # ------------------------------------------------------------------
    def _on_method_changed(self, method: str):
        method = method.lower()

        self._window.kmeansParamsGroup.setVisible("k-means" in method)
        self._window.regionParamsGroup.setVisible("region" in method)
        self._window.meanShiftParamsGroup.setVisible(
            "mean" in method and "shift" in method
        )
        self._window.agglomerativeParamsGroup.setVisible("agglomerative" in method)
        if "region" in method:
            self._window.segInputTitleLbl.setText(
                "Input Image  (click to place seeds)"
            )
        else:
            self._window.segInputTitleLbl.setText("Input Image")

        self._window.segBtnClearSeeds.setEnabled("region" in method)
        self._update_apply_state()

    # ------------------------------------------------------------------
    def _update_apply_state(self):
        if self._image is None:
            self._window.segBtnApply.setEnabled(False)
            return

        method = self._window.segMethodCombo.currentText().lower()
        if "region" in method:
            self._window.segBtnApply.setEnabled(len(self._seeds) > 0)
        else:
            self._window.segBtnApply.setEnabled(True)

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
        self._window.segLegendCanvas.setText("")
        self._window.segLegendCanvas.setPixmap(QPixmap())

        name = Path(path).name
        self._window.segImageNameLbl.setText(name)
        self._update_apply_state()
        self._window.segBtnSave.setEnabled(False)
        self._set_status(f"Loaded: {name}")

    # ------------------------------------------------------------------
    def _on_canvas_click(self, x: int, y: int):
        if self._image is None:
            return
        method = self._window.segMethodCombo.currentText().lower()
        if "region" not in method:
            self._set_status("Seeds are only used with Region Growing.")
            return
        self._seeds.append((x, y))

        # Update list widget
        item = QListWidgetItem(f"Seed {len(self._seeds)}: ({x}, {y})")
        self._window.segSeedsList.addItem(item)

        self._refresh_input_canvas()
        self._update_apply_state()
        self._set_status(
            f"{len(self._seeds)} seed(s) placed — ready to apply."
        )

    # ------------------------------------------------------------------
    def _clear_seeds(self):
        self._seeds = []
        self._window.segSeedsList.clear()
        self._update_apply_state()
        self._refresh_input_canvas()
        self._set_status("Seeds cleared.")

    # ------------------------------------------------------------------
    def _apply(self):
        if self._image is None:
            return

        method = self._window.segMethodCombo.currentText()
        method_l = method.lower()

        if "region" in method_l and not self._seeds:
            self._set_status("Please add at least one seed for Region Growing.",
                             error=True)
            return

        threshold = self._window.regionThreshSpin.value()
        km_clusters = self._window.kmClustersSpin.value()
        ms_spatial_radius = self._window.msSpatialSpin.value()
        ms_color_radius = self._window.msColorSpin.value()
        ms_max_iter = self._window.msIterSpin.value()
        aggl_clusters = self._window.agglClustersSpin.value()
        aggl_block_size = self._window.agglBlockSizeSpin.value()
        aggl_alpha = float(self._window.agglAlphaSpin.value())

        self._window.segBtnApply.setEnabled(False)
        self._window.segBtnLoad.setEnabled(False)
        self._set_status(f"⏳ Processing {method}…")

        self._worker = _SegWorker(
            self._image,
            method,
            self._seeds,
            threshold,
            km_clusters,
            ms_spatial_radius,
            ms_color_radius,
            ms_max_iter,
            aggl_clusters,
            aggl_block_size,
            aggl_alpha,
        )
        self._worker.progress.connect(self._set_status)
        self._worker.finished.connect(self._on_result)
        self._worker.start()

    # ------------------------------------------------------------------
    def _on_result(self, result):
        self._window.segBtnLoad.setEnabled(True)
        self._update_apply_state()

        error_msg = None
        if isinstance(result, tuple) and len(result) == 3:
            kind, payload, error_msg = result

            if kind == "error":
                self._set_status(
                    f"❌ Segmentation failed: {error_msg}", error=True
                )
                return

            if kind == "kmeans":
                self._result = payload["segmented"]
                _show_bgr_on_canvas(self._result, self._window.segOutputCanvas)
                _draw_kmeans_legend(
                    payload["centroids"],
                    payload["labels"],
                    self._window.segLegendCanvas,
                    is_color=self._image.ndim == 3,
                )
                self._window.kmInfoLbl.setText(
                    f"k={payload['k']}  |  "
                    f"Iterations: {payload['iterations']}  |  "
                    f"Inertia: {payload['inertia']:.1f}"
                )
                self._window.segBtnSave.setEnabled(True)
                self._set_status(
                    f"✅ K-Means done — {payload['k']} clusters, "
                    f"{payload['iterations']} iter(s), "
                    f"inertia={payload['inertia']:.1f}"
                )
                return

            # generic result (region growing, mean shift)
            result = payload

        if result is None:
            if error_msg:
                self._set_status(f"❌ Segmentation failed: {error_msg}", error=True)
            else:
                self._set_status("❌ Segmentation failed.", error=True)
            return

        self._result = result
        _show_bgr_on_canvas(result, self._window.segOutputCanvas)
        self._window.segLegendCanvas.setText("")
        self._window.segLegendCanvas.setPixmap(QPixmap())
        self._window.segBtnSave.setEnabled(True)
        method = self._window.segMethodCombo.currentText()
        if "region" in method.lower():
            self._set_status(
                f"✅ Done — {len(self._seeds)} region(s) grown, "
                f"threshold = {self._window.regionThreshSpin.value()}"
            )
        elif "mean" in method.lower() and "shift" in method.lower():
            self._set_status(
                "✅ Done — Mean Shift segmentation complete "
                f"(spatial={self._window.msSpatialSpin.value()}, "
                f"color={self._window.msColorSpin.value()}, "
                f"iter={self._window.msIterSpin.value()})"
            )
        elif "agglomerative" in method.lower():
            self._set_status(
                "✅ Done — Agglomerative segmentation complete "
                f"(clusters={self._window.agglClustersSpin.value()}, "
                f"block={self._window.agglBlockSizeSpin.value()}, "
                f"alpha={self._window.agglAlphaSpin.value():.2f})"
            )
        else:
            self._set_status("✅ Done.")

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


def _draw_kmeans_legend(
    centroids: np.ndarray,
    labels: np.ndarray,
    label,
    is_color: bool = True,
) -> None:
    k   = len(centroids)
    W   = max(label.width(), 300)
    H   = max(label.height(), max(k * 36 + 16, 50))

    pix = QPixmap(W, H)
    pix.fill(QColor("#0a0a0a"))
    painter = QPainter(pix)
    painter.setFont(QFont("Arial", 9))

    total   = labels.size
    sw, sh  = 28, 22
    margin  = 8
    row_h   = sh + 10

    for idx, centroid in enumerate(centroids):
        y = margin + idx * row_h
        if is_color and centroid.shape[0] >= 3:
            b, g, r = int(centroid[0]), int(centroid[1]), int(centroid[2])
        else:
            r = g = b = int(centroid[0])

        painter.setBrush(QBrush(QColor(r, g, b)))
        painter.setPen(QPen(QColor("#888888"), 1))
        painter.drawRect(margin, y, sw, sh)

        pct = 100.0 * np.sum(labels == idx) / total
        val = f"RGB({r},{g},{b})" if is_color else f"Gray({r})"
        painter.setPen(QPen(QColor("#dddddd"), 1))
        painter.drawText(margin + sw + 8, y + sh - 5,
                         f"Cluster {idx + 1}  {val}  {pct:.1f}%")

    painter.end()
    label.setPixmap(pix)