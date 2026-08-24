"""Dependency-light data flow for live pose processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .render import draw_2d_pose, render_3d_pose


LANDMARK_COLUMNS = 4


def landmarks_to_array(landmarks: Any) -> np.ndarray | None:
    """Convert MediaPipe Tasks or legacy landmarks to ``[x, y, z, visibility]``.

    MediaPipe Tasks returns a direct list, while older releases wrapped the
    list in a ``.landmark`` attribute.  Supporting both keeps the pure data
    layer easy to test and tolerant of SDK upgrades.
    """

    if landmarks is None:
        return None
    values = getattr(landmarks, "landmark", landmarks)
    rows: list[tuple[float, float, float, float]] = []
    for landmark in values:
        visibility = getattr(landmark, "visibility", 1.0)
        if visibility is None:
            visibility = 1.0
        coordinates = []
        for name in ("x", "y", "z"):
            value = getattr(landmark, name, None)
            coordinates.append(float("nan") if value is None else float(value))
        rows.append((*coordinates, float(visibility)))
    if not rows:
        return None
    return np.ascontiguousarray(rows, dtype=np.float32)


def _validate_landmark_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != LANDMARK_COLUMNS:
        raise ValueError(f"{name} must have shape [landmarks, 4], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name} cannot be empty")
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class PoseObservation:
    """One person's image-normalized and hip-centered world landmarks."""

    image_landmarks: np.ndarray
    world_landmarks: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "image_landmarks",
            _validate_landmark_array("image_landmarks", self.image_landmarks),
        )
        object.__setattr__(
            self,
            "world_landmarks",
            _validate_landmark_array("world_landmarks", self.world_landmarks),
        )
        if self.image_landmarks.shape[0] != self.world_landmarks.shape[0]:
            raise ValueError("Image and world landmark counts must match")


@dataclass(frozen=True)
class ProcessedFrame:
    """A matched pair of display-ready BGR frames."""

    video_bgr: np.ndarray
    skeleton_bgr: np.ndarray
    pose_found: bool


class PoseDetector(Protocol):
    def process(self, frame_rgb: np.ndarray) -> PoseObservation | None:
        """Infer one pose from an RGB frame."""

    def close(self) -> None:
        """Release detector resources."""


class FrameProcessor:
    """Mirror, infer, and render a camera frame without owning the camera."""

    def __init__(
        self,
        detector: PoseDetector,
        *,
        mirror: bool = True,
        skeleton_width: int = 640,
        skeleton_height: int = 480,
    ) -> None:
        if skeleton_width < 64 or skeleton_height < 64:
            raise ValueError("Skeleton canvas must be at least 64 x 64 pixels")
        self.detector = detector
        self.mirror = mirror
        self.skeleton_width = skeleton_width
        self.skeleton_height = skeleton_height

    def process(self, frame_bgr: np.ndarray) -> ProcessedFrame:
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Camera frame must be a uint8 BGR image with shape [H, W, 3]")

        display_bgr = np.ascontiguousarray(frame[:, ::-1] if self.mirror else frame)
        detector_rgb = np.ascontiguousarray(display_bgr[:, :, ::-1])
        observation = self.detector.process(detector_rgb)

        if observation is None:
            return ProcessedFrame(
                video_bgr=display_bgr,
                skeleton_bgr=render_3d_pose(
                    None,
                    width=self.skeleton_width,
                    height=self.skeleton_height,
                ),
                pose_found=False,
            )

        video_bgr = draw_2d_pose(display_bgr, observation.image_landmarks)
        skeleton_bgr = render_3d_pose(
            observation.world_landmarks,
            width=self.skeleton_width,
            height=self.skeleton_height,
        )
        return ProcessedFrame(
            video_bgr=np.ascontiguousarray(video_bgr),
            skeleton_bgr=np.ascontiguousarray(skeleton_bgr),
            pose_found=True,
        )
