"""PyQt window for camera pose display and portable dataset selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QImage, QKeyEvent, QPixmap, QResizeEvent
from PyQt5.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ai_trainer.runtime_paths import save_dataset_path

from .core import ProcessedFrame
from .worker import CameraConfig, CameraPoseWorker


class ImagePanel(QLabel):
    """Aspect-preserving image label with safe QImage memory ownership."""

    def __init__(self, placeholder: str, parent=None) -> None:
        super().__init__(placeholder, parent)
        self._image: QImage | None = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(420, 320)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #181c25; color: #9aa5b5; "
            "border: 1px solid #3b4352; border-radius: 8px; }"
        )

    def set_bgr_frame(self, frame_bgr: np.ndarray) -> None:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Display frame must be uint8 BGR [H, W, 3]")
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, int(rgb.strides[0]), QImage.Format_RGB888)
        self._image = image.copy()
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image is not None:
            self.setPixmap(
                QPixmap.fromImage(self._image).scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()


def _panel(title: str, image: ImagePanel) -> QGroupBox:
    group = QGroupBox(title)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(10, 16, 10, 10)
    layout.addWidget(image)
    return group


class LivePoseWindow(QMainWindow):
    """Live pose display with a persistent AI Hub dataset folder selector."""

    def __init__(
        self,
        model_path: str | Path,
        config: CameraConfig,
        *,
        dataset_path: Path | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("AI Trainer - Live 3D Pose")
        self.resize(1320, 760)
        self.dataset_path = dataset_path
        self._closing = False
        self._error_shown = False

        self.camera_panel = ImagePanel("Camera is starting…")
        self.skeleton_panel = ImagePanel("3D skeleton is starting…")
        self.status_label = QLabel("Initializing…")
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.dataset_label = QLabel()
        self.dataset_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.dataset_button = QPushButton("Select dataset folder")
        self.dataset_button.setToolTip("Select the extracted AI Hub dataset folder")
        self.dataset_button.clicked.connect(self._choose_dataset)
        self._set_dataset_label(dataset_path)

        header = QHBoxLayout()
        header.addWidget(self.status_label, 1)
        header.addWidget(self.fps_label)

        dataset_row = QHBoxLayout()
        dataset_row.addWidget(self.dataset_button)
        dataset_row.addWidget(self.dataset_label, 1)

        views = QHBoxLayout()
        views.setSpacing(12)
        views.addWidget(_panel("Camera / 2D pose", self.camera_panel), 1)
        views.addWidget(_panel("Estimated 3D skeleton", self.skeleton_panel), 1)

        hint = QLabel(
            "Select the extracted AI Hub dataset folder once. The selection is saved "
            "in your user settings and is used by the reference-data builder. "
            "Press Esc or close the window to stop the camera."
        )
        hint.setObjectName("hint")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addLayout(header)
        layout.addLayout(dataset_row)
        layout.addLayout(views, 1)
        layout.addWidget(hint)
        self.setCentralWidget(central)
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #11151d; color: #e7ecf4; }"
            "QGroupBox { border: 1px solid #343c4b; border-radius: 9px; "
            "margin-top: 8px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }"
            "QLabel#hint { color: #8f9aaa; padding-top: 4px; }"
            "QPushButton { background: #2f6fed; border: 0; border-radius: 5px; padding: 7px 12px; }"
            "QPushButton:hover { background: #4c83ee; }"
        )

        self.worker = CameraPoseWorker(model_path, config, self)
        self.worker.frame_ready.connect(self._update_frame)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.fatal_error.connect(self._show_error)
        self.worker.finished.connect(self._worker_finished)

    def _set_dataset_label(self, dataset_path: Path | None) -> None:
        if dataset_path is None:
            self.dataset_label.setText("Dataset: not selected")
            self.dataset_label.setToolTip("No dataset folder has been selected")
            return
        resolved = Path(dataset_path).expanduser().resolve()
        self.dataset_label.setText(f"Dataset: {resolved}")
        self.dataset_label.setToolTip(str(resolved))

    @pyqtSlot()
    def _choose_dataset(self) -> None:
        initial_directory = str(self.dataset_path or Path.home())
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select extracted AI Hub dataset folder",
            initial_directory,
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        if not selected:
            return
        try:
            self.dataset_path = Path(selected).expanduser().resolve()
            save_dataset_path(self.dataset_path)
        except (OSError, ValueError) as error:
            QMessageBox.warning(self, "Dataset folder", f"Could not save dataset setting:\n{error}")
            return
        self._set_dataset_label(self.dataset_path)
        self.status_label.setText("Dataset folder saved for reference building")
        self.status_label.setStyleSheet("color: #72df8d; font-weight: 600;")

    def start(self) -> None:
        if not self.worker.isRunning():
            self.worker.start()

    @pyqtSlot(object, float)
    def _update_frame(self, frame: ProcessedFrame, fps: float) -> None:
        self.camera_panel.set_bgr_frame(frame.video_bgr)
        self.skeleton_panel.set_bgr_frame(frame.skeleton_bgr)
        self.fps_label.setText(f"{fps:4.1f} FPS")
        if frame.pose_found:
            self.status_label.setText("Person pose detected")
            self.status_label.setStyleSheet("color: #72df8d; font-weight: 600;")
        else:
            self.status_label.setText("Looking for a person pose…")
            self.status_label.setStyleSheet("color: #f2bd61;")

    @pyqtSlot(str)
    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        self.status_label.setStyleSheet("color: #ff7b7b;")
        if not self._error_shown and not self._closing:
            self._error_shown = True
            QMessageBox.critical(self, "Live pose analysis error", message)

    @pyqtSlot()
    def _worker_finished(self) -> None:
        if not self._closing and not self._error_shown:
            self.status_label.setText("Camera stopped")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        if self.worker.isRunning():
            self.worker.requestInterruption()
            if not self.worker.wait(5000):
                self.status_label.setText("Waiting for camera shutdown…")
                event.ignore()
                QTimer.singleShot(250, self.close)
                return
        event.accept()


__all__ = ["ImagePanel", "LivePoseWindow"]
