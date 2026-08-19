"""MediaPipe-based squat pose extraction and measurement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mediapipe as mp

from src.geometry import joint_angle


@dataclass(frozen=True)
class PoseMeasurement:
    left_knee_angle: float
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float


class PoseTracker:
    """Wrap MediaPipe Pose and expose squat-relevant measurements."""

    def __init__(self, min_detection_confidence: float = 0.6) -> None:
        self._pose_module = mp.solutions.pose
        self._pose = self._pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.6,
        )
        self._drawing = mp.solutions.drawing_utils

    def process(self, rgb_frame: Any) -> Any:
        return self._pose.process(rgb_frame)

    @property
    def pose_module(self) -> Any:
        """Expose MediaPipe's pose landmark enum and connection definitions."""
        return self._pose_module

    def draw(self, frame: Any, results: Any) -> None:
        if results.pose_landmarks:
            self._drawing.draw_landmarks(
                frame, results.pose_landmarks, self._pose_module.POSE_CONNECTIONS
            )

    def measure(self, results: Any) -> PoseMeasurement | None:
        if not results.pose_landmarks:
            return None
        landmarks = results.pose_landmarks.landmark
        landmark = self._pose_module.PoseLandmark
        point = lambda name: landmarks[name.value]
        return PoseMeasurement(
            left_knee_angle=joint_angle(point(landmark.LEFT_HIP), point(landmark.LEFT_KNEE), point(landmark.LEFT_ANKLE)),
            right_knee_angle=joint_angle(point(landmark.RIGHT_HIP), point(landmark.RIGHT_KNEE), point(landmark.RIGHT_ANKLE)),
            left_hip_angle=joint_angle(point(landmark.LEFT_SHOULDER), point(landmark.LEFT_HIP), point(landmark.LEFT_KNEE)),
            right_hip_angle=joint_angle(point(landmark.RIGHT_SHOULDER), point(landmark.RIGHT_HIP), point(landmark.RIGHT_KNEE)),
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseTracker":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
