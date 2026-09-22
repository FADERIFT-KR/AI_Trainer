"""MediaPipe Tasks adapter for one-person pose tracking."""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from .core import PoseObservation, landmarks_to_array


class PoseBackendError(RuntimeError):
    """Raised when MediaPipe or its model cannot be initialized."""


class MediaPipePoseDetector:
    """Synchronous VIDEO-mode Pose Landmarker, intended for a worker thread."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model = Path(model_path).expanduser().resolve()
        if not model.is_file() or model.stat().st_size == 0:
            raise PoseBackendError(
                f"MediaPipe pose model not found: {model}. "
                "Run: python scripts/download_pose_model.py"
            )
        for label, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_presence_confidence", min_presence_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{label} must be between 0 and 1")

        try:
            if "MPLCONFIGDIR" not in os.environ:
                matplotlib_cache = Path(tempfile.gettempdir()) / "ai_trainer_matplotlib"
                matplotlib_cache.mkdir(parents=True, exist_ok=True)
                os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)
            import mediapipe as mp
        except ImportError as error:
            raise PoseBackendError(
                "MediaPipe is not installed. Run: python -m pip install -r requirements.txt"
            ) from error

        try:
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_presence_confidence,
                min_tracking_confidence=min_tracking_confidence,
                output_segmentation_masks=False,
            )
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        except (AttributeError, RuntimeError, ValueError) as error:
            raise PoseBackendError(f"Could not initialize MediaPipe Pose Landmarker: {error}") from error
        self._mp = mp
        self._last_timestamp_ms = -1
        self._closed = False

    def _timestamp_ms(self) -> int:
        timestamp = time.monotonic_ns() // 1_000_000
        timestamp = max(timestamp, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        return timestamp

    def process(self, frame_rgb: np.ndarray) -> PoseObservation | None:
        if self._closed:
            raise PoseBackendError("Pose detector is already closed")
        image_array = np.asarray(frame_rgb)
        if image_array.ndim != 3 or image_array.shape[2] != 3 or image_array.dtype != np.uint8:
            raise ValueError("MediaPipe input must be uint8 RGB [H, W, 3]")
        image_array = np.ascontiguousarray(image_array)
        media_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=image_array,
        )
        result: Any = self._landmarker.detect_for_video(media_image, self._timestamp_ms())
        if not result.pose_landmarks or not result.pose_world_landmarks:
            return None
        image_landmarks = landmarks_to_array(result.pose_landmarks[0])
        world_landmarks = landmarks_to_array(result.pose_world_landmarks[0])
        if image_landmarks is None or world_landmarks is None:
            return None
        return PoseObservation(image_landmarks, world_landmarks)

    def close(self) -> None:
        if not self._closed:
            self._landmarker.close()
            self._closed = True

    def __enter__(self) -> "MediaPipePoseDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["MediaPipePoseDetector", "PoseBackendError"]
