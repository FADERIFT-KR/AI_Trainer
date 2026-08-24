"""PyQt5 window for matched live camera and 3-D pose views."""

from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSlot
from PyQt5.QtGui import QCloseEvent, QImage, QKeyEvent, QPixmap, QResizeEvent
from PyQt5.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

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
        image = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        )
        self._image = image.copy()
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._image is None:
            return
        self.setPixmap(
            QPixmap.fromImage(self._image).scaled(
                self.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
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
    def __init__(self, model_path, config: CameraConfig) -> None:
        super().__init__()
        self.setWindowTitle("AI Trainer · 실시간 3D 스켈레톤")
        self.resize(1320, 720)

        self.camera_panel = ImagePanel("카메라 준비 중…")
        self.skeleton_panel = ImagePanel("3D 스켈레톤 준비 중…")
        self.status_label = QLabel("초기화 중…")
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header = QHBoxLayout()
        header.addWidget(self.status_label, 1)
        header.addWidget(self.fps_label)

        views = QHBoxLayout()
        views.setSpacing(12)
        views.addWidget(_panel("실시간 카메라 · 2D 관절", self.camera_panel), 1)
        views.addWidget(_panel("추정 3D 스켈레톤", self.skeleton_panel), 1)

        hint = QLabel(
            "전신이 화면에 들어오도록 카메라에서 2–3 m 떨어져 서세요. "
            "종료: ESC 또는 창 닫기"
        )
        hint.setObjectName("hint")

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addLayout(header)
        layout.addLayout(views, 1)
        layout.addWidget(hint)
        self.setCentralWidget(central)
        self.setStyleSheet(
            "QMainWindow, QWidget { background: #11151d; color: #e7ecf4; }"
            "QGroupBox { border: 1px solid #343c4b; border-radius: 9px; "
            "margin-top: 8px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }"
            "QLabel#hint { color: #8f9aaa; padding-top: 4px; }"
        )

        self.worker = CameraPoseWorker(model_path, config, self)
        self.worker.frame_ready.connect(self._update_frame)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.fatal_error.connect(self._show_error)
        self.worker.finished.connect(self._worker_finished)
        self._closing = False
        self._error_shown = False

    def start(self) -> None:
        if not self.worker.isRunning():
            self.worker.start()

    @pyqtSlot(object, float)
    def _update_frame(self, frame: ProcessedFrame, fps: float) -> None:
        self.camera_panel.set_bgr_frame(frame.video_bgr)
        self.skeleton_panel.set_bgr_frame(frame.skeleton_bgr)
        self.fps_label.setText(f"{fps:4.1f} FPS")
        if frame.pose_found:
            self.status_label.setText("● 자세 감지됨")
            self.status_label.setStyleSheet("color: #72df8d; font-weight: 600;")
        else:
            self.status_label.setText("○ 전신 자세를 찾는 중…")
            self.status_label.setStyleSheet("color: #f2bd61;")

    @pyqtSlot(str)
    def _show_error(self, message: str) -> None:
        self.status_label.setText(f"오류: {message}")
        self.status_label.setStyleSheet("color: #ff7b7b;")
        if not self._error_shown and not self._closing:
            self._error_shown = True
            QMessageBox.critical(self, "실시간 자세 분석 오류", message)

    @pyqtSlot()
    def _worker_finished(self) -> None:
        if not self._closing and not self._error_shown:
            self.status_label.setText("카메라가 종료되었습니다.")

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
                self.status_label.setText("카메라 종료를 기다리는 중…")
                event.ignore()
                QTimer.singleShot(250, self.close)
                return
        event.accept()


__all__ = ["ImagePanel", "LivePoseWindow"]
