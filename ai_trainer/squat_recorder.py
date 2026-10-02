"""Minimal webcam recorder for three squat views, ten repetitions each."""
from __future__ import annotations

import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ai_trainer.camera_devices import CameraDevice, default_camera_index, discover_camera_devices
from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector


REPETITIONS_PER_VIEW = 10
VIEWS = (
    ("front", "정면", "카메라를 정면으로 보고 스쿼트하세요."),
    ("left_oblique", "왼쪽 사선", "인물의 왼쪽 어깨, 골반, 무릎이 보이도록 약 45도 돌세요."),
    ("left_side", "왼쪽 측면", "앞 시점과 같은 방향으로 더 돌아 완전한 측면을 보이세요."),
)
POSE_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "pose_landmarker_full.task"


def _knee_angle(landmarks, width: int, height: int) -> float | None:
    angles = []
    for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
        points = landmarks[[hip, knee, ankle]]
        if min(points[:, 3]) < 0.45:
            continue
        pixels = points[:, :2] * np.array([width, height], dtype=float)
        upper = pixels[0] - pixels[1]
        lower = pixels[2] - pixels[1]
        length = float(np.linalg.norm(upper) * np.linalg.norm(lower))
        if length <= 1e-6:
            continue
        cosine = float(np.dot(upper, lower) / length)
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return float(np.median(angles)) if angles else None


class SquatRepCounter:
    """Count a repetition after a deep knee bend and a stable return to standing."""

    def __init__(
        self,
        descent_angle_deg: float = 135.0,
        standing_angle_deg: float = 145.0,
        stable_frames: int = 2,
    ):
        self.descent_angle_deg = descent_angle_deg
        self.standing_angle_deg = standing_angle_deg
        self.stable_frames = stable_frames
        self.repetitions = 0
        self._phase = "waiting_stand"
        self._standing_frames = 0
        self._descent_frames = 0
        self._return_frames = 0

    def update(self, landmarks, width: int, height: int) -> bool:
        angle = _knee_angle(landmarks, width, height) if landmarks is not None else None
        if angle is None:
            self._standing_frames = 0
            self._descent_frames = 0
            self._return_frames = 0
            return False
        if self._phase == "waiting_stand":
            self._standing_frames = self._standing_frames + 1 if angle >= self.standing_angle_deg else 0
            if self._standing_frames >= self.stable_frames:
                self._phase = "standing"
            return False
        if self._phase == "standing":
            self._descent_frames = self._descent_frames + 1 if angle <= self.descent_angle_deg else 0
            if self._descent_frames >= self.stable_frames:
                self._phase = "bottom_reached"
                self._return_frames = 0
            return False
        self._return_frames = self._return_frames + 1 if angle >= self.standing_angle_deg else 0
        if self._return_frames < self.stable_frames:
            return False
        self.repetitions += 1
        self._phase = "standing"
        self._descent_frames = 0
        self._return_frames = 0
        return True


class SquatRecorderWindow(QMainWindow):
    def __init__(self, camera_index: int | None = None):
        super().__init__()
        self.setWindowTitle("스쿼트 3시점 녹화")
        self.resize(1100, 760)

        camera_index = default_camera_index() if camera_index is None else camera_index
        self._camera_index = camera_index
        self._camera_devices = discover_camera_devices()
        self._capture = self._open_camera(camera_index)
        if self._capture is None:
            raise RuntimeError(f"카메라 {camera_index}번을 열 수 없습니다.")
        try:
            self._pose_detector = MediaPipePoseDetector(POSE_MODEL_PATH)
        except RuntimeError:
            self._capture.release()
            raise

        self._temporary = tempfile.TemporaryDirectory(prefix="squat_recording_")
        self._temporary_path = Path(self._temporary.name)
        self._view_index = 0
        self._repetition_count = 0
        self._rep_counter = SquatRepCounter()
        self._recording = False
        self._writer = None
        self._saved_videos: dict[str, Path] = {}
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not 5.0 <= self._fps <= 60.0:
            self._fps = 30.0

        self.preview = QLabel("카메라 화면을 불러오는 중…")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(720, 405)
        self.preview.setStyleSheet("background: #101820; color: #d8e0e8; font-size: 18px;")

        self.view_label = QLabel()
        self.view_label.setAlignment(Qt.AlignCenter)
        self.view_label.setStyleSheet("font-size: 27px; font-weight: 700; color: #f0f4f8;")
        self.instruction_label = QLabel()
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 17px; color: #b7c4d0;")
        self.count_label = QLabel()
        self.count_label.setAlignment(Qt.AlignCenter)
        self.count_label.setStyleSheet("font-size: 24px; font-weight: 700; color: #78d6a5;")
        self.status_label = QLabel("촬영을 시작할 준비가 되면 버튼을 누르세요.")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 15px; color: #b7c4d0;")

        self.action_button = QPushButton("촬영 시작")
        self.action_button.setMinimumHeight(54)
        self.action_button.setStyleSheet(
            "QPushButton { background: #137c68; color: white; border: 0; border-radius: 6px; "
            "font-size: 18px; font-weight: 700; padding: 8px 24px; }"
            "QPushButton:hover { background: #16977e; }"
        )
        self.action_button.clicked.connect(self._on_action)
        self.camera_button = QPushButton()
        self.camera_button.setMinimumHeight(44)
        self.camera_button.setStyleSheet(
            "QPushButton { background: #33414b; color: #e2e8ed; border: 1px solid #56636d; "
            "border-radius: 6px; font-size: 15px; padding: 8px 16px; }"
            "QPushButton:hover:enabled { background: #40515d; }"
            "QPushButton:disabled { color: #7d8991; }"
        )
        self.camera_button.clicked.connect(self._show_camera_menu)
        self._update_camera_button()
        self.save_button = QPushButton("영상 저장")
        self.save_button.setMinimumHeight(54)
        self.save_button.setEnabled(False)
        self.save_button.setStyleSheet(
            "QPushButton { background: #315f9b; color: white; border: 0; border-radius: 6px; "
            "font-size: 18px; font-weight: 700; padding: 8px 24px; }"
            "QPushButton:hover:enabled { background: #3e75bd; }"
        )
        self.save_button.clicked.connect(self._save_videos)

        buttons = QHBoxLayout()
        buttons.addWidget(self.camera_button)
        buttons.addStretch(1)
        buttons.addWidget(self.action_button)
        buttons.addWidget(self.save_button)
        layout = QVBoxLayout()
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        layout.addWidget(self.view_label)
        layout.addWidget(self.instruction_label)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.count_label)
        layout.addWidget(self.status_label)
        layout.addLayout(buttons)
        root = QWidget()
        root.setLayout(layout)
        root.setStyleSheet("background: #18232b;")
        self.setCentralWidget(root)

        self._update_view_text()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._read_frame)
        self._timer.start(30)

    @staticmethod
    def _open_camera(camera_index: int):
        capture = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(camera_index)
        if not capture.isOpened():
            capture.release()
            return None
        return capture

    def _update_camera_button(self) -> None:
        device = next(
            (item for item in self._camera_devices if item.index == self._camera_index),
            CameraDevice(self._camera_index, "선택된 장치", verified=False),
        )
        self.camera_button.setText(f"카메라 선택 · {device.button_label}")
        self.camera_button.setToolTip(device.device_name or device.description)

    def _show_camera_menu(self) -> None:
        if self._recording:
            return
        self._camera_devices = discover_camera_devices()
        menu = QMenu(self)
        for device in self._camera_devices:
            action = menu.addAction(device.button_label)
            action.setCheckable(True)
            action.setChecked(device.index == self._camera_index)
            action.setToolTip(device.device_name or device.description)
            action.triggered.connect(
                lambda _checked=False, index=device.index: self._select_camera(index)
            )
        menu.exec_(self.camera_button.mapToGlobal(self.camera_button.rect().bottomLeft()))

    def _select_camera(self, camera_index: int) -> None:
        if self._recording or camera_index == self._camera_index:
            return
        replacement = self._open_camera(camera_index)
        if replacement is None:
            QMessageBox.warning(self, "카메라 선택", f"카메라 {camera_index}번을 열 수 없습니다.")
            return
        previous = self._capture
        self._capture = replacement
        self._camera_index = camera_index
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if not 5.0 <= self._fps <= 60.0:
            self._fps = 30.0
        previous.release()
        self._update_camera_button()
        self.status_label.setText(f"카메라 {camera_index}로 변경했습니다.")

    def _update_view_text(self) -> None:
        if self._view_index >= len(VIEWS):
            self.view_label.setText("3개 시점 촬영 완료")
            self.instruction_label.setText("정면, 왼쪽 사선, 왼쪽 측면 영상이 준비됐습니다.")
            self.count_label.setText("각 시점 10회 완료")
            self.action_button.setText("촬영 완료")
            self.action_button.setEnabled(False)
            self.camera_button.setEnabled(False)
            self.save_button.setEnabled(True)
            return
        _, label, instruction = VIEWS[self._view_index]
        self.action_button.setText("촬영 시작")
        self.view_label.setText(f"{self._view_index + 1} / {len(VIEWS)} · {label}")
        self.instruction_label.setText(instruction)
        self.count_label.setText(f"{self._repetition_count} / {REPETITIONS_PER_VIEW}회")

    def _read_frame(self) -> None:
        ok, frame = self._capture.read()
        if not ok:
            self.status_label.setText("카메라 프레임을 읽지 못했습니다.")
            return
        if self._recording:
            if self._writer is None:
                key, _, _ = VIEWS[self._view_index]
                height, width = frame.shape[:2]
                path = self._temporary_path / f"{key}.mp4"
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (width, height),
                )
                if not writer.isOpened():
                    writer.release()
                    self._recording = False
                    self.camera_button.setEnabled(True)
                    self.action_button.setEnabled(True)
                    self.status_label.setText("영상 파일을 만들 수 없습니다. 코덱을 확인해 주세요.")
                    self.action_button.setText("촬영 시작")
                    return
                self._writer = writer
            self._writer.write(frame)
            observation = self._pose_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self._rep_counter.update(
                observation.image_landmarks if observation is not None else None,
                width=frame.shape[1], height=frame.shape[0],
            ):
                self._repetition_count = self._rep_counter.repetitions
                self.count_label.setText(f"{self._repetition_count} / {REPETITIONS_PER_VIEW}회")
                if self._repetition_count >= REPETITIONS_PER_VIEW:
                    self._finish_current_view()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        image = QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.preview.setPixmap(pixmap)

    def _on_action(self) -> None:
        if self._view_index >= len(VIEWS):
            return
        if not self._recording:
            self._recording = True
            self._rep_counter = SquatRepCounter()
            self._repetition_count = 0
            self.count_label.setText(f"0 / {REPETITIONS_PER_VIEW}회")
            self.camera_button.setEnabled(False)
            self.action_button.setText("자동 카운트 녹화 중")
            self.action_button.setEnabled(False)
            self.status_label.setText("서 있는 자세에서 시작하세요. 충분히 내려갔다가 다시 서면 자동으로 셉니다.")

    def _finish_current_view(self) -> None:
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        key, label, _ = VIEWS[self._view_index]
        video_path = self._temporary_path / f"{key}.mp4"
        if not video_path.is_file() or video_path.stat().st_size == 0:
            self._repetition_count = 0
            self.camera_button.setEnabled(True)
            self.action_button.setEnabled(True)
            self.status_label.setText("녹화 영상이 비어 있습니다. 촬영을 다시 시작해 주세요.")
            self.action_button.setText("촬영 시작")
            self._update_view_text()
            return
        self._saved_videos[key] = video_path
        self.action_button.setEnabled(True)
        self._view_index += 1
        self._repetition_count = 0
        self.camera_button.setEnabled(True)
        self._update_view_text()
        if self._view_index < len(VIEWS):
            self.status_label.setText(f"{label} 영상 저장 완료. 다음 시점 촬영을 시작하세요.")
        else:
            self.status_label.setText("모든 시점 촬영이 끝났습니다. 영상 저장을 누르세요.")

    def _save_videos(self) -> None:
        if self._view_index < len(VIEWS) or len(self._saved_videos) != len(VIEWS):
            QMessageBox.information(self, "영상 저장", "세 시점 촬영을 모두 완료한 뒤 저장할 수 있습니다.")
            return
        parent = QFileDialog.getExistingDirectory(self, "스쿼트 영상 저장 폴더 선택", str(Path.home()))
        if not parent:
            return
        base = Path(parent) / datetime.now().strftime("squat_recording_%Y%m%d_%H%M%S")
        destination = base
        suffix = 1
        while destination.exists():
            destination = Path(f"{base}_{suffix:03d}")
            suffix += 1
        try:
            destination.mkdir(parents=True)
            for key, _, _ in VIEWS:
                shutil.copy2(self._saved_videos[key], destination / f"{key}.mp4")
        except OSError as error:
            QMessageBox.warning(self, "저장 실패", f"영상 저장 중 오류가 발생했습니다.\n{error}")
            return
        self.status_label.setText(f"영상 저장 완료: {destination}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self._timer.stop()
        self._recording = False
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._capture.release()
        self._pose_detector.close()
        self._temporary.cleanup()
        super().closeEvent(event)


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication([])
    app.setApplicationName("스쿼트 3시점 녹화")
    try:
        window = SquatRecorderWindow()
    except RuntimeError as error:
        QMessageBox.critical(None, "카메라 오류", str(error))
        return 1
    window.show()
    return int(app.exec_())


__all__ = ["REPETITIONS_PER_VIEW", "SquatRecorderWindow", "SquatRepCounter", "VIEWS", "_knee_angle", "main"]
