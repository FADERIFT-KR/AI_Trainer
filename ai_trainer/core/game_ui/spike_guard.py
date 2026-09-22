"""Causal, symmetric hip/knee outlier guard for live 3D landmarks.

The 1-euro filter handles small jitter but increases its cutoff for a sudden
large observation.  A one-frame tracking swap after occlusion can therefore
pass almost unchanged.  This guard holds the first implausible jump; a
persistent, anatomically plausible change is followed with a bounded step.
"""
from __future__ import annotations

import numpy as np

from ai_trainer.core.common_skeleton import COMMON_JOINT_NAMES


_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_HIPS = (_IDX["LHip"], _IDX["RHip"])
_KNEES = ((_IDX["LKnee"], _IDX["LHip"], _IDX["LAnkle"]),
          (_IDX["RKnee"], _IDX["RHip"], _IDX["RAnkle"]))
_FEET = ((_IDX["LAnkle"], _IDX["LKnee"], _IDX["LHeel"], _IDX["LBigToe"]),
         (_IDX["RAnkle"], _IDX["RKnee"], _IDX["RHeel"], _IDX["RBigToe"]))
_PELVIS = _IDX["Hip"]


class Live3DSpikeGuard:
    """Reject isolated large landmark innovations before adaptive smoothing."""

    def __init__(self, hip_step_fraction: float = 0.045, knee_step_fraction: float = 0.08,
                 ankle_step_fraction: float = 0.13, stabilize_feet: bool = False):
        self.hip_step_fraction = hip_step_fraction
        self.knee_step_fraction = knee_step_fraction
        self.ankle_step_fraction = ankle_step_fraction
        self.stabilize_feet = stabilize_feet
        self.last = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
        self.has_last = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)
        self.pending = np.zeros_like(self.last)
        self.pending_count = np.zeros(len(COMMON_JOINT_NAMES), dtype=np.int32)

    def _leg_scale(self, pose: np.ndarray) -> float:
        lengths = []
        for knee, hip, ankle in _KNEES:
            if self.has_last[[knee, hip, ankle]].all():
                lengths.append(np.linalg.norm(self.last[knee] - self.last[hip])
                               + np.linalg.norm(self.last[ankle] - self.last[knee]))
            else:
                lengths.append(np.linalg.norm(pose[knee] - pose[hip])
                               + np.linalg.norm(pose[ankle] - pose[knee]))
        return float(np.clip(np.median(lengths), 0.4, 1.5))

    def _process_joint(self, out: np.ndarray, good: np.ndarray, old: np.ndarray,
                       joint: int, reference: int | None, max_step: float,
                       structurally_bad: bool = False) -> bool:
        if not good[joint] or not np.isfinite(out[joint]).all():
            self.pending_count[joint] = 0
            if self.has_last[joint]:
                out[joint] = old[joint]
            return False
        if not self.has_last[joint]:
            self.last[joint] = out[joint]
            self.has_last[joint] = True
            return False

        base = out[reference] if reference is not None else np.zeros(3)
        old_base = old[reference] if reference is not None else np.zeros(3)
        candidate = out[joint] - base
        previous = old[joint] - old_base
        delta = candidate - previous
        distance = float(np.linalg.norm(delta))
        if distance <= max_step:
            self.pending_count[joint] = 0
            self.last[joint] = out[joint]
            return False

        if self.pending_count[joint] and np.linalg.norm(candidate - self.pending[joint]) <= max_step:
            self.pending_count[joint] += 1
        else:
            self.pending_count[joint] = 1
        self.pending[joint] = candidate

        if self.pending_count[joint] < 2 or structurally_bad:
            out[joint] = base + previous
        else:
            out[joint] = base + previous + delta * (max_step / distance)
        self.last[joint] = out[joint]
        return True

    def __call__(self, pose: np.ndarray, good_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        out = np.asarray(pose, dtype=np.float64).copy()
        good = np.asarray(good_mask, dtype=bool)
        old = self.last.copy()
        guarded = np.zeros(len(COMMON_JOINT_NAMES), dtype=bool)
        scale = self._leg_scale(out)

        for hip in _HIPS:
            guarded[hip] = self._process_joint(out, good, old, hip, None,
                                                self.hip_step_fraction * scale)
        out[_PELVIS] = (out[_HIPS[0]] + out[_HIPS[1]]) / 2.0
        self.last[_PELVIS] = out[_PELVIS]
        self.has_last[_PELVIS] = self.has_last[list(_HIPS)].all()

        for knee, hip, ankle in _KNEES:
            bad_length = False
            if self.has_last[[knee, hip, ankle]].all() and good[[knee, hip, ankle]].all():
                old_thigh = float(np.linalg.norm(old[knee] - old[hip]))
                old_shank = float(np.linalg.norm(old[ankle] - old[knee]))
                new_thigh = float(np.linalg.norm(out[knee] - out[hip]))
                new_shank = float(np.linalg.norm(out[ankle] - out[knee]))
                bad_length = (abs(new_thigh - old_thigh) > 0.25 * max(old_thigh, 1e-6)
                              or abs(new_shank - old_shank) > 0.25 * max(old_shank, 1e-6))
            guarded[knee] = self._process_joint(out, good, old, knee, hip,
                                                 self.knee_step_fraction * scale, bad_length)

        if not self.stabilize_feet:
            for _, _, ankle in _KNEES:
                if good[ankle] and np.isfinite(out[ankle]).all():
                    self.last[ankle] = out[ankle]
                    self.has_last[ankle] = True
            return out, guarded

        for ankle, knee, heel, toe in _FEET:
            observed_ankle = out[ankle].copy()
            guarded[ankle] = self._process_joint(
                out, good, old, ankle, knee, self.ankle_step_fraction * scale
            )
            # MediaPipe feet often jump together in camera depth. Keep the
            # heel/toe attached to the corrected ankle instead of stretching
            # the rendered foot into an impossible shape.
            correction = out[ankle] - observed_ankle
            if good[ankle]:
                for foot_joint in (heel, toe):
                    if good[foot_joint]:
                        out[foot_joint] += correction
                        guarded[foot_joint] = guarded[ankle]
        return out, guarded


class DisplayLegLengthStabilizer:
    """Project displayed leg segments to a stable person-specific length.

    MediaPipe's camera-depth estimate can shorten a shin at occlusion even
    when its 2D endpoints remain visible. This is display-only; it must not
    feed the DTW classifier, which was trained on its own coordinate domain.
    """

    def __init__(self, calibration_frames: int = 8, max_rotation_radians: float = 0.28):
        self.calibration_frames = calibration_frames
        self.max_rotation_radians = max_rotation_radians
        self.samples: dict[tuple[int, int], list[float]] = {}
        self.directions: dict[tuple[int, int], np.ndarray] = {}

    def __call__(self, pose: np.ndarray, confidence: np.ndarray,
                 frozen: np.ndarray, min_visibility: float) -> np.ndarray:
        out = np.asarray(pose, dtype=np.float64).copy()
        for knee, hip, ankle in _KNEES:
            side = "L" if knee == _IDX["LKnee"] else "R"
            heel, toe = _IDX[f"{side}Heel"], _IDX[f"{side}BigToe"]
            for parent, child, descendants in (
                (hip, knee, (ankle, heel, toe)),
                (knee, ankle, (heel, toe)),
            ):
                segment = out[child] - out[parent]
                length = float(np.linalg.norm(segment))
                if not np.isfinite(length) or length < 1e-6:
                    continue
                key = (parent, child)
                samples = self.samples.setdefault(key, [])
                if (len(samples) < self.calibration_frames
                        and min(confidence[parent], confidence[child]) >= min_visibility
                        and not frozen[parent] and not frozen[child]
                        and 0.15 <= length <= 0.75):
                    samples.append(length)
                if not samples:
                    continue
                target = float(np.median(samples))
                direction = segment / length
                previous = self.directions.get(key)
                if previous is not None:
                    angle = float(np.arccos(np.clip(np.dot(previous, direction), -1.0, 1.0)))
                    if angle > self.max_rotation_radians:
                        fraction = self.max_rotation_radians / angle
                        if np.sin(angle) > 1e-5:
                            direction = (np.sin((1.0 - fraction) * angle) * previous
                                         + np.sin(fraction * angle) * direction) / np.sin(angle)
                        else:
                            direction = (1.0 - fraction) * previous + fraction * direction
                            direction /= max(float(np.linalg.norm(direction)), 1e-8)
                self.directions[key] = direction
                correction = direction * target - segment
                out[child] += correction
                for joint in descendants:
                    out[joint] += correction
        return out
