from __future__ import annotations

from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import QMainWindow, QMessageBox

from controllers.face_controller import FaceDetectionController


class AppController(QObject):
    """Main application controller (Face Detection App)."""

    def __init__(self, window: QMainWindow):
        super().__init__(window)
        self.window = window

        # ── Load UI ─────────────────────────────
        ui_path = Path(__file__).resolve().parents[1] / "ui" / "main.ui"
        uic.loadUi(str(ui_path), self.window)

        # ── Initialize Face Detection Controller ─
        self.face_controller = FaceDetectionController(self.window)
        self.face_controller.bind_ui(self.window)

        # ── Menu actions ────────────────────────
        if hasattr(self.window, "actionQuit"):
            self.window.actionQuit.triggered.connect(self.window.close)

        if hasattr(self.window, "actionAbout"):
            self.window.actionAbout.triggered.connect(self._about)

        # ── Signals ────────────────────────────
        self.face_controller.status_message.connect(self.window.statusbar.showMessage)
        self.face_controller.error_occurred.connect(self._show_error)

    # --------------------------------------------------
    def _show_error(self, msg: str):
        QMessageBox.critical(self.window, "Error", msg)

    # --------------------------------------------------
    def _about(self):
        QMessageBox.about(
            self.window,
            "About",
            "<b>Face Detection App</b><br><br>"
            "Detects faces using Haar Cascade.<br>"
            "Built with OpenCV + PyQt5.",
        )