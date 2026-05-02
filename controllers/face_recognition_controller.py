from __future__ import annotations

import traceback
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import QObject, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QLabel, QMainWindow

from core.face.classifier import FaceClassifier
from core.face.process_orl import IMG_SIZE


class FaceRecognitionController(QObject):

    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    _BTN_LOAD = "faceRecBtnLoad"
    _BTN_RECOGNIZE = "faceRecBtnRecognize"

    _LBL_INPUT = "faceRecInputCanvas"
    _LBL_OUTPUT = "faceRecOutputCanvas"
    _LBL_NAME = "faceRecImageNameLbl"
    _LBL_STATUS = "faceRecStatusLbl"

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.window = window
        self._image: Optional[np.ndarray] = None

        # model
        self._classifier: Optional[FaceClassifier] = None

    def bind_ui(self, window: QMainWindow):
        self._w = window
        self._widget(self._BTN_LOAD).clicked.connect(self._on_load)
        self._widget(self._BTN_RECOGNIZE).clicked.connect(self._on_recognize)

        self._widget(self._BTN_RECOGNIZE).setEnabled(False)

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "Open Image", "", "Images (*.png *.jpg *.jpeg *.pgm)"
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            return

        self._image = img
        self._show(img, self._LBL_INPUT)
        self._widget(self._LBL_NAME).setText(Path(path).name)
        self._widget(self._BTN_RECOGNIZE).setEnabled(True)

    def _on_recognize(self):
        try:
            if self._image is None:
                return

            self.status_message.emit("Preparing dataset and model…")
            self._ensure_model()

            # preprocess input image
            gray = cv2.cvtColor(self._image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, IMG_SIZE)
            norm = resized.astype(np.float32) / 255.0
            vec = norm.flatten()

            match = self._classifier.predict(vec)

            match_path = match.image_path
            matched_img = cv2.imread(str(match_path))
            if matched_img is None:
                matched_img = cv2.cvtColor(cv2.imread(str(match_path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)

            self._show(matched_img, self._LBL_OUTPUT)
            self._widget(self._LBL_STATUS).setText(
                f"Matched: {match_path.name} (subject {match.subject_id}, distance {match.distance:.4f})"
            )
            self.status_message.emit("Recognition complete")

        except Exception as e:
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def _ensure_model(self):
        if self._classifier is not None:
            return

        model_path = Path(__file__).resolve().parents[2] / "face-and-segmentation-toolkit/pca_model.pkl"
        self._classifier = FaceClassifier(str(model_path)).load()


        
    def _widget(self, name):
        return self.window.findChild(object, name)

    def _show(self, img, widget_name: str):
        label: QLabel = self._widget(widget_name)

        h, w = img.shape[:2]
        # ensure 3 channels for display
        if len(img.shape) == 2 or img.shape[2] == 1:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio))
