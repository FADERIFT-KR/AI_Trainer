"""Camera-plane guidance for the *displayed* monocular 3D skeleton.

MediaPipe world landmarks can place a side-view ankle above the knee during
deep flexion even while the image landmarks show the correct ordering. Keep
the 3D lateral coordinate, but guide the visible sagittal projection with
the filtered image landmarks. The classifier never receives this output.
When a side-view arm is occluded, mirror the visible arm about the torso for
display only; an invisible MediaPipe wrist is not a reliable 3D observation.
"""
from __future__ import annotations

import numpy as np

from ai_trainer.camera_views import VIEW_LEFT, VIEW_RIGHT
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES


_IDX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
_LEG_JOINTS = tuple(_IDX[f"{side}{name}"] for side in "LR"
                    for name in ("Hip", "Knee", "Ankle", "Heel", "BigToe"))


class ImageGuidedSkeletonDisplay:
    """Apply image-plane leg guidance and visibility-aware arm display poses."""

    def __init__(self, view: str, calibration_frames: int = 8, image_weight: float = 0.9):
        self.view = view
        self.calibration_frames = calibration_frames
        self.image_weight = image_weight
        self._leg_pixel_lengths: list[float] = []
        self._last_near_wrist_offset: float | None = None
        self._last_near_wrist_image: np.ndarray | None = None

    def update(self, common_2d: np.ndarray, aligned_3d: np.ndarray | None,
               world_landmarks: np.ndarray | None = None) -> np.ndarray | None:
        points = np.asarray(common_2d, dtype=float)
        if points.shape != (len(COMMON_JOINT_NAMES), 2):
            raise ValueError("common_2d must contain one (x,y) point per common joint")
        if self.view not in (VIEW_LEFT, VIEW_RIGHT):
            return aligned_3d
        side = "L" if self.view == VIEW_LEFT else "R"
        hip, knee, ankle = (_IDX[f"{side}{name}"] for name in ("Hip", "Knee", "Ankle"))
        leg_length = (np.linalg.norm(points[hip] - points[knee])
                      + np.linalg.norm(points[knee] - points[ankle]))
        if (len(self._leg_pixel_lengths) < self.calibration_frames and np.isfinite(leg_length)
                and leg_length >= 50):
            self._leg_pixel_lengths.append(float(leg_length))
        if aligned_3d is None:
            return aligned_3d
        aligned = np.asarray(aligned_3d, dtype=float)
        if aligned.shape != (len(COMMON_JOINT_NAMES), 3):
            raise ValueError("aligned_3d must have shape (18,3)")
        result = aligned.copy()
        if self._leg_pixel_lengths:
            scale = float(np.median(self._leg_pixel_lengths))
            pelvis = points[_IDX["Hip"]]
            projected_x = (points[:, 0] - pelvis[0]) / scale
            projected_y = (pelvis[1] - points[:, 1]) / scale
            sign = -1.0 if self.view == VIEW_RIGHT else 1.0
            for joint in _LEG_JOINTS:
                if np.isfinite(points[joint]).all():
                    result[joint, 2] = ((1.0 - self.image_weight) * aligned[joint, 2]
                                        + self.image_weight * sign * projected_x[joint])
                    result[joint, 1] = ((1.0 - self.image_weight) * aligned[joint, 1]
                                        + self.image_weight * projected_y[joint])
        if world_landmarks is not None:
            landmarks = np.asarray(world_landmarks, dtype=float)
            if landmarks.shape != (33, 4):
                raise ValueError("world_landmarks must have shape (33,4)")
            self._stabilize_visible_wrist_depth(result, points, landmarks[:, 3], side)
            self._guide_occluded_arm(result, landmarks[:, 3], side)
        return result

    def _stabilize_visible_wrist_depth(self, result: np.ndarray, points: np.ndarray,
                                       visibility: np.ndarray, near_side: str) -> None:
        """Limit an unobserved lateral 3D leap when the image wrist barely moves."""
        shoulder = _IDX[f"{near_side}Shoulder"]
        wrist = _IDX[f"{near_side}Wrist"]
        wrist_mp = 15 if near_side == "L" else 16
        if (not np.isfinite(visibility[wrist_mp]) or visibility[wrist_mp] < 0.65
                or not np.isfinite(result[[shoulder, wrist]]).all()):
            return
        offset = float(result[wrist, 0] - result[shoulder, 0])
        image_point = points[wrist]
        if (self._last_near_wrist_offset is not None and self._last_near_wrist_image is not None
                and np.isfinite(image_point).all()
                and np.linalg.norm(image_point - self._last_near_wrist_image) < 25.0):
            offset = float(np.clip(offset, self._last_near_wrist_offset - 0.12,
                                   self._last_near_wrist_offset + 0.12))
            result[wrist, 0] = result[shoulder, 0] + offset
        self._last_near_wrist_offset = offset
        self._last_near_wrist_image = image_point.copy()

    @staticmethod
    def _guide_occluded_arm(result: np.ndarray, visibility: np.ndarray, near_side: str) -> None:
        """Replace unreliable far-arm segments with mirrored near-arm segments.

        The correction is deliberately limited to rendering. Visibility is
        blended over a range so a wrist does not snap when confidence crosses
        a single threshold. Mirroring body-aligned x preserves segment length.
        """
        far_side = "R" if near_side == "L" else "L"
        mp_indices = {"L": (11, 13, 15), "R": (12, 14, 16)}
        near_shoulder, near_elbow, near_wrist = (
            _IDX[f"{near_side}{name}"] for name in ("Shoulder", "Elbow", "Wrist")
        )
        far_shoulder, far_elbow, far_wrist = (
            _IDX[f"{far_side}{name}"] for name in ("Shoulder", "Elbow", "Wrist")
        )
        _, near_elbow_mp, near_wrist_mp = mp_indices[near_side]
        _, far_elbow_mp, far_wrist_mp = mp_indices[far_side]

        def mirror_offset(start: int, end: int) -> np.ndarray:
            offset = result[end] - result[start]
            return offset * np.array([-1.0, 1.0, 1.0])

        for target, origin, near_start, near_end, far_mp, near_mp in (
            (far_elbow, far_shoulder, near_shoulder, near_elbow, far_elbow_mp, near_elbow_mp),
            (far_wrist, far_elbow, near_elbow, near_wrist, far_wrist_mp, near_wrist_mp),
        ):
            if (not np.isfinite(visibility[near_mp]) or visibility[near_mp] < 0.65
                    or not np.isfinite(result[[origin, near_start, near_end]]).all()):
                continue
            predicted = result[origin] + mirror_offset(near_start, near_end)
            if not np.isfinite(predicted).all():
                continue
            # Full replacement below 0.45; transition to observed coordinates
            # above 0.75. No change is made to a confidently observed arm.
            observed_weight = (float(np.clip((visibility[far_mp] - 0.45) / 0.30, 0.0, 1.0))
                               if np.isfinite(visibility[far_mp]) else 0.0)
            if not np.isfinite(result[target]).all():
                observed_weight = 0.0
            result[target] = observed_weight * result[target] + (1.0 - observed_weight) * predicted
