"""Diagnostic MediaPipe-to-AI-Hub body-proportion adapter.

This module is intentionally not wired into the production pipeline.  It changes
segment lengths while preserving their per-frame directions; it never targets a
joint angle or a posture class.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

_I = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


@dataclass(frozen=True)
class GeometryTargets:
    hip_width_over_torso: float = 0.3679116141572338
    shoulder_width_over_torso: float = 0.6180796258857161
    left_thigh_over_torso: float = 0.5529784834164009
    right_thigh_over_torso: float = 0.5553409930995847
    left_shin_over_torso: float = 0.4088437292669934
    right_shin_over_torso: float = 0.4132830399856944


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-8)


def adapt_mediapipe_to_training_domain(
    sequence_2d: np.ndarray,
    targets: GeometryTargets = GeometryTargets(),
    *,
    enabled: bool = True,
) -> np.ndarray:
    """Align body proportions without changing temporal order or bone directions."""
    src = np.asarray(sequence_2d)
    if src.ndim != 3 or src.shape[1:] != (18, 2):
        raise ValueError("sequence_2d must have shape (T, 18, 2)")
    if not enabled:
        return src.copy()
    out = src.astype(np.float64, copy=True)
    for frame in out:
        hip = frame[_I["Hip"]].copy()
        neck = frame[_I["Neck"]].copy()
        torso = max(float(np.linalg.norm(neck - hip)), 1e-8)

        hip_axis = _unit(frame[_I["RHip"]] - frame[_I["LHip"]])
        half_hip = targets.hip_width_over_torso * torso / 2.0
        old_lhip, old_rhip = frame[_I["LHip"]].copy(), frame[_I["RHip"]].copy()
        frame[_I["LHip"]] = hip - hip_axis * half_hip
        frame[_I["RHip"]] = hip + hip_axis * half_hip

        shoulder_axis = _unit(frame[_I["RShoulder"]] - frame[_I["LShoulder"]])
        half_shoulder = targets.shoulder_width_over_torso * torso / 2.0
        for side in ("L", "R"):
            shoulder = f"{side}Shoulder"
            delta = neck + shoulder_axis * (half_shoulder if side == "R" else -half_shoulder) - frame[_I[shoulder]]
            for name in (shoulder, f"{side}Elbow", f"{side}Wrist"):
                frame[_I[name]] += delta

        for side, thigh_ratio, shin_ratio, old_hip in (
            ("L", targets.left_thigh_over_torso, targets.left_shin_over_torso, old_lhip),
            ("R", targets.right_thigh_over_torso, targets.right_shin_over_torso, old_rhip),
        ):
            knee, ankle = f"{side}Knee", f"{side}Ankle"
            thigh_dir = _unit(frame[_I[knee]] - old_hip)
            shin_dir = _unit(frame[_I[ankle]] - frame[_I[knee]])
            old_ankle = frame[_I[ankle]].copy()
            frame[_I[knee]] = frame[_I[f"{side}Hip"]] + thigh_dir * thigh_ratio * torso
            frame[_I[ankle]] = frame[_I[knee]] + shin_dir * shin_ratio * torso
            foot_delta = frame[_I[ankle]] - old_ankle
            frame[_I[f"{side}Heel"]] += foot_delta
            frame[_I[f"{side}BigToe"]] += foot_delta
    if not np.isfinite(out).all():
        raise ValueError("adapter produced non-finite coordinates")
    return out.astype(src.dtype, copy=False)


__all__ = ["GeometryTargets", "adapt_mediapipe_to_training_domain"]
