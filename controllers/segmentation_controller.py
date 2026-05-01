"""
controllers/segmentation_tab_controller.py
-------------------------------------------
Unified controller for the Segmentation tab.

Manages the combo-box switch between:
  • K-Means        (index 0) — Lolo's algorithm
  • Region Growing (index 1) — Roro's algorithm

All shared widgets (Load, Apply, Save, canvases, status, legend strip)
live once in the .ui; this controller owns them and routes work to the
correct algorithm based on the current combo selection.

The seed-click logic (for Region Growing) is also handled here so
it can be activated/deactivated cleanly when switching methods.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import QFileDialog, QListWidgetItem, QMainWindow

# ── algorithms ────────────────────────────────────────────────────────────────
from core.segmentation.kmeans import apply_kmeans_segmentation
from core.segmentation.region_growing import region_growing

METHOD_KMEANS  = 0
METHOD_REGION  = 1


# ─────────────────────────────────────────────────────────────────────────────
# Workers
# ─────────────────────────────────────────────────────────────────────────────

def _kmeans_worker_fn(image: np.ndarray, k: int, queue: mp.Queue) -> None:
    try:
        queue.put(("kmeans", apply_kmeans_segmentation(image, k=k, random_state=42)))
    except Exception as exc:
        queue.put(("error", str(exc)))


def _region_worker_fn(image: np.ndarray,
                      seeds: list[tuple[int, int]],
                      threshold: int,
                      queue: mp.Queue) -> None:
    try:
        result = region_growing(image, seeds, threshold)
        queue.put(("region", result))
    except Exception as exc:
        queue.put(("error", str(exc)))


class _SegWorker(QThread):
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
# Clickable canvas helper (Region Growing seeds)
# ─────────────────────────────────────────────────────────────────────────────

class _ClickableCanvas:
    """Monkey-patches a QLabel to report image-space clicks."""

    def __init__(self, label, callback):
        self._label    = label
        self._callback = callback
        self._img_w    = 1
        self._img_h    = 1
        self._active   = False
        label.mousePressEvent = self._on_click

    def set_image_size(self, w: int, h: int):
        self._img_w, self._img_h = w, h

    def set_active(self, active: bool):
        self._active = active

    def _on_click(self, event):
        if not self._active:
            return
        lw, lh = self._label.width(), self._label.height()
        if lw == 0 or lh == 0:
            return
        scale  = min(lw / self._img_w, lh / self._img_h)
        disp_w = int(self._img_w * scale)
        disp_h = int(self._img_h * scale)
        off_x  = (lw - disp_w) // 2
        off_y  = (lh - disp_h) // 2
        px = event.x() - off_x
        py = event.y() - off_y
        if 0 <= px < disp_w and 0 <= py < disp_h:
            self._callback(int(px / scale), int(py / scale))


# ─────────────────────────────────────────────────────────────────────────────
# Unified Controller
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationTabController(QObject):
    status_message = pyqtSignal(str)

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self._window     = window
        self._image      : np.ndarray | None = None
        self._result_img : np.ndarray | None = None
        self._seeds      : list[tuple[int, int]] = []
        self._worker     : _SegWorker | None = None

        self._canvas = _ClickableCanvas(
            window.segInputCanvas, self._on_canvas_click
        )

    # ------------------------------------------------------------------ bind
    def bind_ui(self):
        w = self._window
        w.segBtnLoad.clicked.connect(self._load_image)
        w.segBtnApply.clicked.connect(self._apply)
        w.segBtnSave.clicked.connect(self._save)
        w.segBtnClearSeeds.clicked.connect(self._clear_seeds)
        w.segMethodCombo.currentIndexChanged.connect(self._on_method_changed)

        # Show correct params group on startup
        self._on_method_changed(w.segMethodCombo.currentIndex())

    # --------------------------------------------------------- method switch
    def _on_method_changed(self, index: int):
        w = self._window
        is_kmeans = (index == METHOD_KMEANS)
        is_region = (index == METHOD_REGION)

        w.kmeansParamsGroup.setVisible(is_kmeans)
        w.regionParamsGroup.setVisible(is_region)

        # Seed clicks only active for Region Growing
        self._canvas.set_active(is_region)

        # Update input canvas title hint
        if is_region:
            w.segInputTitleLbl.setText("Input Image  (click to place seeds)")
        else:
            w.segInputTitleLbl.setText("Input Image")

        # Reset stale outputs
        self._result_img = None
        self._seeds      = []
        w.segSeedsList.clear()
        w.segOutputCanvas.setText("Result will appear here")
        w.segOutputCanvas.setPixmap(QPixmap())
        w.segLegendCanvas.setText("")
        w.segLegendCanvas.setPixmap(QPixmap())
        w.segBtnSave.setEnabled(False)

        # Re-draw input without seed markers
        if self._image is not None:
            _show_bgr(self._image, w.segInputCanvas)
            # K-Means only needs an image to be ready; Region Growing needs seeds too
            w.segBtnApply.setEnabled(is_kmeans)

    # ------------------------------------------------------------------ load
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

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        self._image      = img
        self._result_img = None
        self._seeds      = []
        self._window.segSeedsList.clear()

        h, w = img.shape[:2]
        self._canvas.set_image_size(w, h)

        _show_bgr(img, self._window.segInputCanvas)
        self._window.segOutputCanvas.setText("Result will appear here")
        self._window.segOutputCanvas.setPixmap(QPixmap())
        self._window.segLegendCanvas.setText("")
        self._window.segLegendCanvas.setPixmap(QPixmap())

        name = Path(path).name
        self._window.segImageNameLbl.setText(name)
        self._window.segBtnSave.setEnabled(False)

        method = self._window.segMethodCombo.currentIndex()
        # K-Means is ready as soon as image is loaded; Region Growing needs seeds
        self._window.segBtnApply.setEnabled(method == METHOD_KMEANS)

        self._set_status(
            f"Loaded: {name}"
            + ("  |  Click image to add seed points."
               if method == METHOD_REGION else "")
        )

    # ---------------------------------------------------- seed click (region)
    def _on_canvas_click(self, x: int, y: int):
        if self._image is None:
            return
        self._seeds.append((x, y))

        item = QListWidgetItem(f"Seed {len(self._seeds)}: ({x}, {y})")
        self._window.segSeedsList.addItem(item)

        self._refresh_input_with_seeds()
        self._window.segBtnApply.setEnabled(True)
        self._set_status(f"{len(self._seeds)} seed(s) placed — ready to apply.")

    def _clear_seeds(self):
        self._seeds = []
        self._window.segSeedsList.clear()
        self._window.segBtnApply.setEnabled(False)
        if self._image is not None:
            _show_bgr(self._image, self._window.segInputCanvas)
        self._set_status("Seeds cleared.")

    def _refresh_input_with_seeds(self):
        if self._image is None:
            return
        vis = self._image.copy()
        _SEED_COLORS = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0),
            (0, 200, 255), (255, 0, 200), (255, 200, 0),
        ]
        for idx, (sx, sy) in enumerate(self._seeds):
            bgr = _SEED_COLORS[idx % len(_SEED_COLORS)]
            r   = 8
            cv2.circle(vis, (sx, sy), r, bgr, 2)
            cv2.line(vis, (sx - r, sy), (sx + r, sy), bgr, 2)
            cv2.line(vis, (sx, sy - r), (sx, sy + r), bgr, 2)
            cv2.putText(vis, str(idx + 1), (sx + r + 2, sy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)
        _show_bgr(vis, self._window.segInputCanvas)

    # ----------------------------------------------------------------- apply
    def _apply(self):
        if self._image is None:
            return

        method = self._window.segMethodCombo.currentIndex()

        self._window.segBtnApply.setEnabled(False)
        self._window.segBtnLoad.setEnabled(False)

        if method == METHOD_KMEANS:
            k = self._window.kmClustersSpin.value()
            self._set_status(f"⏳ Running K-Means with k={k}…")
            self._worker = _SegWorker(
                _kmeans_worker_fn, (self._image, k)
            )
        else:
            if not self._seeds:
                self._set_status("⚠️ Place at least one seed first.", error=True)
                self._window.segBtnApply.setEnabled(True)
                self._window.segBtnLoad.setEnabled(True)
                return
            threshold = self._window.regionThreshSpin.value()
            self._set_status("⏳ Growing regions…")
            self._worker = _SegWorker(
                _region_worker_fn, (self._image, self._seeds, threshold)
            )

        self._worker.finished.connect(self._on_result)
        self._worker.start()

    # --------------------------------------------------------------- result
    def _on_result(self, kind: str, payload):
        self._window.segBtnApply.setEnabled(True)
        self._window.segBtnLoad.setEnabled(True)

        if kind == "error":
            self._set_status(f"❌ Error: {payload}", error=True)
            return

        if kind == "kmeans":
            self._result_img = payload["segmented"]
            _show_bgr(self._result_img, self._window.segOutputCanvas)
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
            self._set_status(
                f"✅ K-Means done — {payload['k']} clusters, "
                f"{payload['iterations']} iter(s), "
                f"inertia={payload['inertia']:.1f}"
            )

        elif kind == "region":
            self._result_img = payload
            _show_bgr(self._result_img, self._window.segOutputCanvas)
            self._window.segLegendCanvas.setText("")
            self._window.segLegendCanvas.setPixmap(QPixmap())
            self._set_status(
                f"✅ Region Growing done — "
                f"{len(self._seeds)} region(s) grown, "
                f"threshold={self._window.regionThreshSpin.value()}"
            )

        self._window.segBtnSave.setEnabled(True)

    # ------------------------------------------------------------------ save
    def _save(self):
        if self._result_img is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self._window, "Save Result", "segmentation_result.png",
            "PNG (*.png);;JPEG (*.jpg)"
        )
        if path:
            cv2.imwrite(path, self._result_img)
            self._set_status(f"💾 Saved to {Path(path).name}")

    # ---------------------------------------------------------------- helpers
    def _set_status(self, msg: str, error: bool = False):
        colour = "#cc4444" if error else "#88cc88"
        self._window.segStatusLbl.setStyleSheet(
            f"color:{colour}; font-size:10pt;"
        )
        self._window.segStatusLbl.setText(msg)
        self.status_message.emit(msg)


# ─────────────────────────────────────────────────────────────────────────────
# Canvas helpers (private to this module)
# ─────────────────────────────────────────────────────────────────────────────

def _show_bgr(img: np.ndarray, label) -> None:
    lw = max(label.width(), 1)
    lh = max(label.height(), 1)
    if img.ndim == 2:
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w = img.shape[:2]
        c    = img.shape[2]
        if c == 4:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        else:
            rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg).scaled(
        lw, lh, Qt.KeepAspectRatio, Qt.SmoothTransformation
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