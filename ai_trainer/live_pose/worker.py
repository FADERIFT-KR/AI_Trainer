"""Qt worker thread that owns the camera and MediaPipe detector."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from .core import FrameProcessor
from .mediapipe_pose import MediaPipePoseDetector, PoseBackendError


@dataclass(frozen=True)
class CameraConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    requested_fps: int = 30
    mirror: bool = True
    skeleton_width: int = 640
    skeleton_height: int = 480
    confidence: float = 0.4  # 실사용 환경(조명/거리 이상적이지 않음)에서 recall을 우선

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera_index cannot be negative")
        if self.width <= 0 or self.height <= 0 or self.requested_fps <= 0:
            raise ValueError("Camera width, height, and FPS must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


def _open_camera(cv2: object, config: CameraConfig):
    backends: list[tuple[str, int]] = []
    if sys.platform == "win32":
        for name in ("CAP_MSMF", "CAP_DSHOW"):
            backend = getattr(cv2, name, None)
            if backend is not None and all(value != backend for _, value in backends):
                backends.append((name, backend))
    any_backend = getattr(cv2, "CAP_ANY", 0)
    if all(value != any_backend for _, value in backends):
        backends.append(("CAP_ANY", any_backend))

    capture = None
    selected_report = None
    for backend_name, backend in backends:
        candidate = cv2.VideoCapture(config.camera_index, backend)
        if not candidate.isOpened():
            candidate.release()
            continue

        set_results = {
            "width": bool(candidate.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)),
            "height": bool(candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)),
            "fps": bool(candidate.set(cv2.CAP_PROP_FPS, config.requested_fps)),
        }
        buffer_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
        set_results["buffersize"] = (
            bool(candidate.set(buffer_property, 1)) if buffer_property is not None else False
        )

        warmup_ms = []
        valid_frame = False
        for _ in range(3):
            started = time.perf_counter()
            success, frame = candidate.read()
            warmup_ms.append((time.perf_counter() - started) * 1000.0)
            if success and frame is not None:
                valid_frame = True
            else:
                valid_frame = False
                break
        if not valid_frame:
            candidate.release()
            continue

        fourcc_value = int(candidate.get(cv2.CAP_PROP_FOURCC))
        fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
        capture = candidate
        selected_report = {
            "requested_backend": "CAP_MSMF" if sys.platform == "win32" else "CAP_ANY",
            "selected_candidate": backend_name,
            "actual_backend": candidate.getBackendName() if hasattr(candidate, "getBackendName") else backend_name,
            "width": float(candidate.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": float(candidate.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "requested_fps": config.requested_fps,
            "reported_fps": float(candidate.get(cv2.CAP_PROP_FPS)),
            "buffersize": float(candidate.get(buffer_property)) if buffer_property is not None else None,
            "fourcc": fourcc,
            "fourcc_value": fourcc_value,
            "set_results": set_results,
            "warmup_capture_ms": warmup_ms,
        }
        break
    if capture is None:
        raise RuntimeError(
            f"카메라 {config.camera_index}을(를) 열 수 없습니다. "
            "Windows 설정에서 카메라 권한과 다른 앱의 사용 여부를 확인하세요."
        )

    _open_camera.last_report = selected_report
    return capture


class CameraPoseWorker(QThread):
    """Capture and infer off the GUI thread; emit matched display frames."""

    frame_ready = pyqtSignal(object, float)
    status_changed = pyqtSignal(str)
    fatal_error = pyqtSignal(str)

    def __init__(
        self,
        model_path: str | Path,
        config: CameraConfig,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.model_path = Path(model_path).resolve()
        self.config = config

    def run(self) -> None:
        capture = None
        detector = None
        try:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError(
                    "OpenCV가 설치되지 않았습니다. "
                    "python -m pip install -r requirements.txt 를 실행하세요."
                ) from error

            self.status_changed.emit("3D 자세 모델을 불러오는 중…")
            detector = MediaPipePoseDetector(
                self.model_path,
                min_detection_confidence=self.config.confidence,
                min_presence_confidence=self.config.confidence,
                min_tracking_confidence=self.config.confidence,
            )
            processor = FrameProcessor(
                detector,
                mirror=self.config.mirror,
                skeleton_width=self.config.skeleton_width,
                skeleton_height=self.config.skeleton_height,
            )

            self.status_changed.emit("카메라를 여는 중…")
            capture = _open_camera(cv2, self.config)
            self.status_changed.emit("카메라 실행 중")

            previous_time = time.perf_counter()
            smoothed_fps = 0.0
            consecutive_failures = 0
            while not self.isInterruptionRequested():
                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        raise RuntimeError("카메라 프레임을 연속으로 읽지 못했습니다.")
                    self.msleep(10)
                    continue
                consecutive_failures = 0
                processed = processor.process(frame_bgr)

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                smoothed_fps = (
                    instantaneous_fps
                    if smoothed_fps == 0.0
                    else smoothed_fps * 0.90 + instantaneous_fps * 0.10
                )
                self.frame_ready.emit(processed, smoothed_fps)
        except (PoseBackendError, RuntimeError, ValueError, OSError) as error:
            self.fatal_error.emit(str(error))
        except Exception as error:  # Media backends can raise vendor-specific errors.
            self.fatal_error.emit(f"실시간 자세 처리 중 예기치 않은 오류: {error}")
        finally:
            if capture is not None:
                capture.release()
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass


__all__ = ["CameraConfig", "CameraPoseWorker"]
