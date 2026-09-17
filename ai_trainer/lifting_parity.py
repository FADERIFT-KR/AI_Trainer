"""Read-only helpers for comparing live and stored offline lifting inputs."""
from __future__ import annotations

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES
from .normalization import body_axes, leg_length_scale

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def live_preprocess_2d(raw_window: np.ndarray, calib_frames: int = 8) -> tuple[np.ndarray, float]:
    """Reproduce OnlineSquatSession's model-input normalization for one window."""
    raw = np.asarray(raw_window, dtype=np.float64)
    n = min(calib_frames, len(raw))
    torso = np.linalg.norm(raw[:n, _IDX["Neck"]] - raw[:n, _IDX["Hip"]], axis=-1)
    scale = max(float(np.median(torso)), 1e-6)
    centered = raw - raw[:, _IDX["Hip"] : _IDX["Hip"] + 1]
    return (centered / scale).astype(np.float32), scale


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1, v2 = a - b, c - b
    cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def mean_leg_angles(coords: np.ndarray) -> tuple[float, float]:
    left_hip = joint_angle(coords[_IDX["Neck"]], coords[_IDX["LHip"]], coords[_IDX["LKnee"]])
    right_hip = joint_angle(coords[_IDX["Neck"]], coords[_IDX["RHip"]], coords[_IDX["RKnee"]])
    left_knee = joint_angle(coords[_IDX["LHip"]], coords[_IDX["LKnee"]], coords[_IDX["LAnkle"]])
    right_knee = joint_angle(coords[_IDX["RHip"]], coords[_IDX["RKnee"]], coords[_IDX["RAnkle"]])
    return (left_hip + right_hip) / 2.0, (left_knee + right_knee) / 2.0


def finalize_single_3d(coords: np.ndarray) -> np.ndarray:
    """Scale/orient one output frame for a deterministic parity-only comparison."""
    scale = max(float(leg_length_scale(coords[None])[0]), 1e-6)
    scaled = coords / scale
    axes = body_axes(scaled)
    return np.einsum("ij,pj->pi", axes.T, scaled)


def pelvis_height(coords: np.ndarray) -> float:
    return float(-(coords[_IDX["LAnkle"], 1] + coords[_IDX["RAnkle"], 1]) / 2.0)


__all__ = [
    "live_preprocess_2d", "mean_leg_angles", "finalize_single_3d", "pelvis_height"
]
