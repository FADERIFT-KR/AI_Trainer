"""Reversible preprocessing for fixed-length, body-centred 3-D skeletons."""
from __future__ import annotations

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.core.s4_normalize.normalization import body_axes
from ai_trainer.squat.phase_features import extract_phase_features
from ai_trainer.squat.phase_segmentation import PhaseBoundaries, segment_phases

from .schema import N_FRAMES, N_JOINTS, NormalizationContext

_PELVIS = COMMON_JOINT_NAMES.index("Hip")
_L_HIP = COMMON_JOINT_NAMES.index("LHip")
_R_HIP = COMMON_JOINT_NAMES.index("RHip")
_L_KNEE = COMMON_JOINT_NAMES.index("LKnee")
_R_KNEE = COMMON_JOINT_NAMES.index("RKnee")
_L_ANKLE = COMMON_JOINT_NAMES.index("LAnkle")
_R_ANKLE = COMMON_JOINT_NAMES.index("RAnkle")


def resample_sequence(coords: np.ndarray, frames: int = N_FRAMES) -> np.ndarray:
    source = np.asarray(coords, dtype=np.float64)
    if source.ndim != 3 or source.shape[1:] != (N_JOINTS, 3):
        raise ValueError(f"coords must have shape (T,{N_JOINTS},3), got {source.shape}")
    if len(source) < 2 or not np.isfinite(source).all():
        raise ValueError("coords must contain at least two finite frames")
    old_t = np.linspace(0.0, 1.0, len(source))
    new_t = np.linspace(0.0, 1.0, frames)
    result = np.empty((frames, N_JOINTS, 3), dtype=np.float64)
    for joint in range(N_JOINTS):
        for axis in range(3):
            result[:, joint, axis] = np.interp(new_t, old_t, source[:, joint, axis])
    return result


def _leg_scale(centered: np.ndarray) -> float:
    left = np.linalg.norm(centered[:, _L_HIP] - centered[:, _L_KNEE], axis=-1)
    left += np.linalg.norm(centered[:, _L_KNEE] - centered[:, _L_ANKLE], axis=-1)
    right = np.linalg.norm(centered[:, _R_HIP] - centered[:, _R_KNEE], axis=-1)
    right += np.linalg.norm(centered[:, _R_KNEE] - centered[:, _R_ANKLE], axis=-1)
    scale = float(np.median((left + right) / 2.0))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("invalid leg-length scale")
    return scale


def _reference_rotation(scaled: np.ndarray, reference_frames: int = 5) -> np.ndarray:
    count = max(1, min(reference_frames, len(scaled)))
    mean_axes = np.stack([body_axes(scaled[i]) for i in range(count)]).mean(axis=0)
    lateral = mean_axes[:, 0] / (np.linalg.norm(mean_axes[:, 0]) + 1e-8)
    forward = np.cross(lateral, mean_axes[:, 1])
    forward /= np.linalg.norm(forward) + 1e-8
    vertical = np.cross(forward, lateral)
    vertical /= np.linalg.norm(vertical) + 1e-8
    return np.stack([lateral, vertical, forward], axis=1)


def normalize_sequence(coords: np.ndarray) -> tuple[np.ndarray, NormalizationContext]:
    """Resample and normalize while retaining everything required to invert it."""
    sampled = resample_sequence(coords)
    pelvis = sampled[:, _PELVIS].copy()
    centered = sampled - pelvis[:, None, :]
    scale = _leg_scale(centered)
    scaled = centered / scale
    rotation = _reference_rotation(scaled)
    aligned = np.einsum("ij,tpj->tpi", rotation.T, scaled)
    return aligned.astype(np.float32), NormalizationContext(pelvis, scale, rotation)


def denormalize_sequence(coords: np.ndarray, context: NormalizationContext) -> np.ndarray:
    aligned = np.asarray(coords, dtype=np.float64)
    if aligned.shape != (N_FRAMES, N_JOINTS, 3):
        raise ValueError("normalized skeleton has an unexpected shape")
    world = np.einsum("ij,tpj->tpi", context.rotation, aligned)
    return world * context.scale + context.pelvis[:, None, :]


def phase_boundaries(coords: np.ndarray) -> PhaseBoundaries:
    return segment_phases(extract_phase_features(np.asarray(coords)))


def smooth_phase_gate(coords: np.ndarray) -> np.ndarray:
    """Return a soft gate covering descent, bottom and ascent."""
    bounds = phase_boundaries(coords)
    gate = np.zeros(N_FRAMES, dtype=np.float64)
    start = max(0, bounds.prep_end)
    end = min(N_FRAMES, bounds.rise_end)
    gate[start:end] = 1.0
    ramp = min(5, max(1, (end - start) // 4))
    if end > start and ramp:
        gate[start : start + ramp] = np.linspace(0.0, 1.0, ramp, endpoint=False)
        gate[end - ramp : end] = np.linspace(1.0, 0.0, ramp, endpoint=False)
    return gate.astype(np.float32)
