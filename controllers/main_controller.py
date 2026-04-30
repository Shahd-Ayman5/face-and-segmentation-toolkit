"""
controllers/main_controller.py
--------------------------------
Main application controller.
Loads the UI and wires up all tab controllers:
  - Face Detection  (existing)
  - Thresholding    (new – Spectral Thresholding)
  - Segmentation    (new – Region Growing)
"""

from __future__ import annotations

from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from controllers.face_controller import FaceDetectionController
from controllers.thresholding_controller import ThresholdingController
from controllers.segmentation_controller import SegmentationController


class AppController(QObject):
    """Root controller – owns one child controller per tab."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.window = window

        # ── Load UI ─────────────────────────────────────────────────
        ui_path = Path(__file__).resolve().parents[1] / "ui" / "main.ui"
        uic.loadUi(str(ui_path), self.window)

        # ── Tab controllers ─────────────────────────────────────────

        # 1. Face detection (unchanged)
        self.face_controller = FaceDetectionController(self.window)
        self.face_controller.bind_ui(self.window)

        # 2. Thresholding (Spectral)
        self.thresh_controller = ThresholdingController(self.window)
        self.thresh_controller.bind_ui()

        # 3. Segmentation (Region Growing)
        self.seg_controller = SegmentationController(self.window)
        self.seg_controller.bind_ui()

        # ── Menu ────────────────────────────────────────────────────
        if hasattr(self.window, "actionQuit"):
            self.window.actionQuit.triggered.connect(self.window.close)

        if hasattr(self.window, "actionAbout"):
            self.window.actionAbout.triggered.connect(self._about)

        # ── Status bar ──────────────────────────────────────────────
        self.face_controller.status_message.connect(
            self.window.statusbar.showMessage
        )
        self.thresh_controller.status_message.connect(
            self.window.statusbar.showMessage
        )
        self.seg_controller.status_message.connect(
            self.window.statusbar.showMessage
        )

        # Error handling (face controller has its own error signal)
        if hasattr(self.face_controller, "error_occurred"):
            self.face_controller.error_occurred.connect(self._show_error)

    # ------------------------------------------------------------------
    def _show_error(self, msg: str):
        QMessageBox.critical(self.window, "Error", msg)

    def _about(self):
        QMessageBox.about(
            self.window,
            "About",
            "<b>Vision Toolkit</b><br><br>"
            "• Face Detection (Haar Cascade)<br>"
            "• Spectral Thresholding (multi-level)<br>"
            "• Region Growing Segmentation<br><br>"
            "Built with OpenCV + PyQt5.",
        )