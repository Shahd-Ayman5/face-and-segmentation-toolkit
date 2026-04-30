from __future__ import annotations

import traceback
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QMainWindow

from core.face.face_detector import detect_faces


class FaceDetectionController(QObject):

    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    _BTN_LOAD = "faceBtnLoad"
    _BTN_RUN  = "faceBtnRun"
    _BTN_SAVE = "faceBtnSave"

    _LBL_CANVAS = "faceCanvasLabel"
    _LBL_NAME   = "faceImageNameLbl"
    _LBL_STATS  = "faceStatsLbl"

    _SPN_SCALE  = "faceSpnScale"
    _SPN_NEIGH  = "faceSpnNeighbors"

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.window = window
        self._image: Optional[np.ndarray] = None
        self._result_vis: Optional[np.ndarray] = None

    def bind_ui(self, window: QMainWindow):
        self._w = window

        self._widget(self._BTN_LOAD).clicked.connect(self._on_load)
        self._widget(self._BTN_RUN).clicked.connect(self._on_run)
        self._widget(self._BTN_SAVE).clicked.connect(self._on_save)

        self._widget(self._BTN_RUN).setEnabled(False)
        self._widget(self._BTN_SAVE).setEnabled(False)

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            return

        self._image = img
        self._show(img)

        self._widget(self._LBL_NAME).setText(Path(path).name)
        self._widget(self._BTN_RUN).setEnabled(True)

    def _on_run(self):
        try:
            scale = self._spin(self._SPN_SCALE, 1.3)
            neigh = int(self._spin(self._SPN_NEIGH, 5))

            result = detect_faces(self._image, scale, neigh)

            self._result_vis = result.visualisation
            self._show(result.visualisation)

            self._widget(self._LBL_STATS).setText(
                f"Faces: {result.num_faces}\nTime: {result.computation_time_ms:.2f} ms"
            )

            self._widget(self._BTN_SAVE).setEnabled(True)

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _on_save(self):
        path, _ = QFileDialog.getSaveFileName(self.window, "Save", "faces.png")
        if path:
            cv2.imwrite(path, self._result_vis)

    def _widget(self, name):
        return self.window.findChild(object, name)

    def _spin(self, name, default):
        w = self._widget(name)
        return w.value() if w else default

    def _show(self, img):
        label: QLabel = self._widget(self._LBL_CANVAS)

        h, w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        qimg = QImage(img_rgb.data, w, h, 3*w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))