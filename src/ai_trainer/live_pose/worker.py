"""Qt worker thread that owns a platform-native camera and MediaPipe detector."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QThread, pyqtSignal

from .core import FrameProcessor
from .mediapipe_pose import MediaPipePoseDetector, PoseBackendError


@dataclass(frozen=True, slots=True)
class CameraConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    requested_fps: int = 30
    mirror: bool = True
    skeleton_width: int = 640
    skeleton_height: int = 480
    confidence: float = 0.5

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera_index cannot be negative")
        if self.width <= 0 or self.height <= 0 or self.requested_fps <= 0:
            raise ValueError("Camera width, height, and FPS must be positive")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")


def _camera_backends(cv2: object, platform_name: str | None = None) -> list[int]:
    """Return usable capture backends in a portable, native-first order."""

    platform_name = sys.platform if platform_name is None else platform_name
    names = ["CAP_ANY"]
    if platform_name.startswith("win"):
        names.extend(("CAP_DSHOW", "CAP_MSMF"))
    elif platform_name == "darwin":
        names.append("CAP_AVFOUNDATION")
    else:
        names.append("CAP_V4L2")
    backends: list[int] = []
    for name in names:
        backend = getattr(cv2, name, None)
        if isinstance(backend, int) and backend not in backends:
            backends.append(backend)
    return backends or [0]


def _open_camera(cv2: object, config: CameraConfig):
    capture = None
    for backend in _camera_backends(cv2):
        candidate = cv2.VideoCapture(config.camera_index, backend)
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()
    if capture is None:
        raise RuntimeError(
            f"Could not open camera {config.camera_index}. Check that it is connected, "
            "not in use by another application, and that this app has camera permission "
            "in your operating-system privacy settings."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    capture.set(cv2.CAP_PROP_FPS, config.requested_fps)
    buffer_property = getattr(cv2, "CAP_PROP_BUFFERSIZE", None)
    if buffer_property is not None:
        capture.set(buffer_property, 1)
    return capture


class CameraPoseWorker(QThread):
    """Capture and infer off the GUI thread; emit matched display frames."""

    frame_ready = pyqtSignal(object, float)
    status_changed = pyqtSignal(str)
    fatal_error = pyqtSignal(str)

    def __init__(self, model_path: str | Path, config: CameraConfig, parent=None) -> None:
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
                    "OpenCV is not installed. Run: python -m pip install -r requirements.txt"
                ) from error

            self.status_changed.emit("Loading 3D pose model…")
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

            self.status_changed.emit("Opening camera…")
            capture = _open_camera(cv2, self.config)
            self.status_changed.emit("Camera is running")

            previous_time = time.perf_counter()
            smoothed_fps = 0.0
            consecutive_failures = 0
            while not self.isInterruptionRequested():
                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        raise RuntimeError("Camera did not provide a frame for 30 consecutive reads")
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
        except Exception as error:
            self.fatal_error.emit(f"Unexpected live-pose processing error: {error}")
        finally:
            if capture is not None:
                capture.release()
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass


__all__ = ["CameraConfig", "CameraPoseWorker", "_camera_backends"]
