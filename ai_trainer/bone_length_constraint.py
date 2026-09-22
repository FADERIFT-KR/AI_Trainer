"""Diagnostic-only subject bone-length constraints for Common Skeleton 3D.

This module is deliberately not connected to the game pipeline.  It keeps each
leg segment's observed direction, fixes its calibrated length, and reconstructs
the ankle/heel/toe triangle with calibrated side lengths.  It does not claim to
recover ground-truth depth or foot contact.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u, v = a - b, c - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom <= 1e-12:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))


def _unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-12:
        return vector / norm
    if fallback is not None:
        fallback_norm = float(np.linalg.norm(fallback))
        if fallback_norm > 1e-12:
            return fallback / fallback_norm
    raise ValueError("Cannot constrain a zero-length bone without a valid fallback direction")


@dataclass(frozen=True)
class SideLengths:
    thigh: float
    shin: float
    ankle_heel: float
    ankle_toe: float
    heel_toe: float


@dataclass(frozen=True)
class BoneLengthCalibration:
    left: SideLengths
    right: SideLengths
    source_indices: tuple[int, ...]


def calibrate_leg_lengths(
    frames: np.ndarray,
    *,
    required_frames: int = 8,
    min_knee_angle_deg: float = 150.0,
) -> BoneLengthCalibration:
    """Calibrate from the first finite, standing frames; never silently use a squat."""
    coords = np.asarray(frames, dtype=float)
    if coords.ndim != 3 or coords.shape[1:] != (18, 3):
        raise ValueError(f"Expected (T,18,3), got {coords.shape}")
    selected: list[int] = []
    for index, frame in enumerate(coords):
        if not np.isfinite(frame).all():
            continue
        knees = [
            _angle_deg(frame[IDX[side + "Hip"]], frame[IDX[side + "Knee"]], frame[IDX[side + "Ankle"]])
            for side in ("L", "R")
        ]
        if np.isfinite(knees).all() and min(knees) >= min_knee_angle_deg:
            selected.append(index)
            if len(selected) == required_frames:
                break
    if len(selected) < required_frames:
        raise ValueError(
            f"Need {required_frames} standing frames with both knee angles >= "
            f"{min_knee_angle_deg:g} degrees; found {len(selected)}"
        )

    sample = coords[selected]

    def side_lengths(side: str) -> SideLengths:
        hip, knee, ankle, heel, toe = (IDX[side + name] for name in ("Hip", "Knee", "Ankle", "Heel", "BigToe"))
        distance = lambda a, b: np.linalg.norm(sample[:, a] - sample[:, b], axis=1)
        values = [
            float(np.median(distance(hip, knee))),
            float(np.median(distance(knee, ankle))),
            float(np.median(distance(ankle, heel))),
            float(np.median(distance(ankle, toe))),
            float(np.median(distance(heel, toe))),
        ]
        if min(values) <= 1e-8:
            raise ValueError(f"Degenerate {side} calibration lengths: {values}")
        # Independent medians can violate the triangle inequality by tiny amounts.
        values[4] = float(np.clip(values[4], abs(values[2] - values[3]) + 1e-8, values[2] + values[3] - 1e-8))
        return SideLengths(*values)

    return BoneLengthCalibration(side_lengths("L"), side_lengths("R"), tuple(selected))


def constrain_leg_lengths(frame: np.ndarray, calibration: BoneLengthCalibration) -> np.ndarray:
    """Return a constrained copy. Hip/root and all non-leg joints stay unchanged."""
    original = np.asarray(frame, dtype=float)
    if original.shape != (18, 3) or not np.isfinite(original).all():
        raise ValueError("Expected finite (18,3) frame")
    result = original.copy()

    for side, lengths in (("L", calibration.left), ("R", calibration.right)):
        hip, knee, ankle, heel, toe = (IDX[side + name] for name in ("Hip", "Knee", "Ankle", "Heel", "BigToe"))
        thigh_direction = _unit(original[knee] - original[hip])
        shin_direction = _unit(original[ankle] - original[knee])
        result[knee] = result[hip] + thigh_direction * lengths.thigh
        result[ankle] = result[knee] + shin_direction * lengths.shin

        heel_direction = _unit(original[heel] - original[ankle])
        toe_direction = _unit(original[toe] - original[ankle])
        perpendicular = toe_direction - np.dot(toe_direction, heel_direction) * heel_direction
        if np.linalg.norm(perpendicular) <= 1e-12:
            # Pick the least parallel Cartesian axis without changing the heel direction.
            axis = np.eye(3)[int(np.argmin(np.abs(heel_direction)))]
            perpendicular = axis - np.dot(axis, heel_direction) * heel_direction
        perpendicular = _unit(perpendicular)
        cosine = (lengths.ankle_heel**2 + lengths.ankle_toe**2 - lengths.heel_toe**2) / (
            2.0 * lengths.ankle_heel * lengths.ankle_toe
        )
        cosine = float(np.clip(cosine, -1.0, 1.0))
        sine = float(np.sqrt(max(0.0, 1.0 - cosine * cosine)))
        result[heel] = result[ankle] + heel_direction * lengths.ankle_heel
        result[toe] = result[ankle] + lengths.ankle_toe * (cosine * heel_direction + sine * perpendicular)

    return result


def constrain_sequence(frames: np.ndarray, calibration: BoneLengthCalibration) -> np.ndarray:
    return np.stack([constrain_leg_lengths(frame, calibration) for frame in np.asarray(frames)])

