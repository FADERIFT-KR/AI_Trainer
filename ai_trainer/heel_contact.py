"""2D image-coordinate evidence for heel contact during a squat."""
from __future__ import annotations

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

_IDX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
_L_HEEL, _R_HEEL = _IDX["LHeel"], _IDX["RHeel"]
_L_BIGTOE, _R_BIGTOE = _IDX["LBigToe"], _IDX["RBigToe"]
_HIP, _NECK = _IDX["Hip"], _IDX["Neck"]


def ema_filter_sequence(coords: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Apply the same short causal EMA used by the live pose session."""
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(COMMON_JOINT_NAMES), 2):
        raise ValueError("2D skeleton sequence shape가 (T,18,2)가 아닙니다.")
    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("EMA alpha는 (0,1] 범위여야 합니다.")
    if len(values) == 0:
        return values.copy()
    filtered = values.copy()
    for index in range(1, len(filtered)):
        filtered[index] = (
            float(alpha) * values[index]
            + (1.0 - float(alpha)) * filtered[index - 1]
        )
    return filtered


def evaluate_heel_contact_2d(
    frames: np.ndarray,
    *,
    scale: float,
    baseline_toe_y: np.ndarray,
    baseline_heel_y: np.ndarray,
    threshold: float = 0.10,
    min_usable_fraction: float = 0.5,
    consecutive_frames: int = 2,
) -> tuple[bool | None, tuple[float, float] | None]:
    """Classify both heels as stable, lifted, or insufficient evidence.

    Image Y grows downward, therefore ``baseline_y-current_y`` is positive
    when a landmark rises. The same-side big toe must remain near its baseline
    so camera/body translation is not confused with an isolated heel lift.
    """
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(COMMON_JOINT_NAMES), 2):
        return None, None
    if len(values) < max(2, int(consecutive_frames)) or not np.isfinite(scale):
        return None, None
    scale = max(float(scale), 1e-6)
    toe_baseline = np.asarray(baseline_toe_y, dtype=np.float64).reshape(2)
    heel_baseline = np.asarray(baseline_heel_y, dtype=np.float64).reshape(2)
    toe_y = values[:, [_L_BIGTOE, _R_BIGTOE], 1]
    heel_y = values[:, [_L_HEEL, _R_HEEL], 1]
    toe_shift = (toe_baseline[None, :] - toe_y) / scale
    heel_lift = (heel_baseline[None, :] - heel_y) / scale

    finite = np.isfinite(toe_shift) & np.isfinite(heel_lift)
    toe_grounded = finite & (np.abs(toe_shift) <= float(threshold))
    required = max(
        int(consecutive_frames),
        int(np.ceil(len(values) * float(min_usable_fraction))),
    )
    if np.any(np.sum(toe_grounded, axis=0) < required):
        return None, None

    max_delta = tuple(
        float(np.max(heel_lift[toe_grounded[:, side], side]))
        for side in range(2)
    )
    lifted = toe_grounded & (heel_lift > float(threshold))
    run = np.ones(2, dtype=np.int32)
    persistent = np.zeros(2, dtype=bool)
    for index in range(len(values)):
        run = np.where(lifted[index], run if index == 0 else run + 1, 0)
        persistent |= run >= int(consecutive_frames)
    if bool(np.any(persistent)):
        return False, max_delta

    stable = toe_grounded & (np.abs(heel_lift) <= float(threshold))
    if bool(np.all(np.sum(stable, axis=0) >= required)):
        return True, max_delta
    return None, max_delta


def estimate_sequence_heel_contact(
    coords_2d: np.ndarray,
    prep_bounds: tuple[int, int] | list[int] | None,
    *,
    calibration_frames: int = 8,
    threshold: float = 0.10,
    ema_alpha: float = 0.45,
) -> tuple[bool | None, tuple[float, float] | None]:
    """Offline proxy for the live countdown heel baseline.

    AIHub clips do not contain the separate UI countdown. Their preparation
    phase (or the first frames when that phase is empty) is used as the fixed
    baseline, then the remainder of the clip is checked identically to live.
    """
    filtered = ema_filter_sequence(coords_2d, ema_alpha)
    if len(filtered) < 3:
        return None, None
    start, end = (0, 0) if prep_bounds is None else map(int, prep_bounds)
    start = max(0, min(start, len(filtered) - 1))
    end = max(start, min(end, len(filtered)))
    if end - start < 2:
        start, end = 0, min(len(filtered), max(2, int(calibration_frames)))
    else:
        end = min(end, start + max(2, int(calibration_frames)))
    baseline = filtered[start:end]
    torso = np.linalg.norm(
        baseline[:, _NECK] - baseline[:, _HIP], axis=1
    )
    finite_torso = torso[np.isfinite(torso) & (torso > 1e-6)]
    if finite_torso.size == 0:
        return None, None
    scale = float(np.median(finite_torso))
    baseline_toe = np.median(
        baseline[:, [_L_BIGTOE, _R_BIGTOE], 1], axis=0
    )
    baseline_heel = np.median(
        baseline[:, [_L_HEEL, _R_HEEL], 1], axis=0
    )
    evaluation = filtered[end:] if len(filtered) - end >= 2 else filtered
    return evaluate_heel_contact_2d(
        evaluation,
        scale=scale,
        baseline_toe_y=baseline_toe,
        baseline_heel_y=baseline_heel,
        threshold=threshold,
    )


__all__ = [
    "ema_filter_sequence",
    "estimate_sequence_heel_contact",
    "evaluate_heel_contact_2d",
]
