"""Label-preserving augmentation for normalized 3D skeleton sequences.

The transforms are intentionally small.  They model differences in repetition
speed, residual camera yaw after orientation normalization, and smooth pose
estimation jitter without changing the semantic squat class.
"""
from __future__ import annotations

import numpy as np

from .common_skeleton import PELVIS_IDX


def temporal_resample(coords: np.ndarray, duration_scale: float) -> np.ndarray:
    """Resample ``(T,J,3)`` coordinates to a scaled sequence duration."""
    values = np.asarray(coords, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("coords must have shape (T, J, 3)")
    if values.shape[0] < 2:
        raise ValueError("at least two frames are required")
    if not np.isfinite(duration_scale) or duration_scale <= 0.0:
        raise ValueError("duration_scale must be positive")

    target_frames = max(10, int(round(values.shape[0] * float(duration_scale))))
    source_time = np.linspace(0.0, 1.0, values.shape[0])
    target_time = np.linspace(0.0, 1.0, target_frames)
    flat = values.reshape(values.shape[0], -1)
    resampled = np.stack(
        [np.interp(target_time, source_time, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return resampled.reshape(target_frames, values.shape[1], 3)


def yaw_rotate(coords: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a hip-centred skeleton around its vertical (Y) axis."""
    values = np.asarray(coords, dtype=np.float64)
    radians = np.deg2rad(float(yaw_deg))
    cosine, sine = np.cos(radians), np.sin(radians)
    rotated = values.copy()
    x, z = values[..., 0], values[..., 2]
    rotated[..., 0] = cosine * x + sine * z
    rotated[..., 2] = -sine * x + cosine * z
    return rotated


def add_smooth_jitter(
    coords: np.ndarray,
    jitter_std: float,
    rng: np.random.Generator,
    *,
    smoothing: float = 0.80,
) -> np.ndarray:
    """Add low-amplitude, temporally correlated landmark noise."""
    values = np.asarray(coords, dtype=np.float64)
    if jitter_std <= 0.0:
        return values.copy()
    smoothing = float(np.clip(smoothing, 0.0, 0.99))
    noise = rng.normal(0.0, float(jitter_std), size=values.shape)
    for frame in range(1, len(noise)):
        noise[frame] = smoothing * noise[frame - 1] + (1.0 - smoothing) * noise[frame]

    # Keep the normalized coordinate convention: the pelvis is the origin in
    # every frame.  Subtracting root noise also makes neighbouring joints move
    # coherently instead of translating the whole body.
    noise -= noise[:, PELVIS_IDX : PELVIS_IDX + 1]
    augmented = values + noise
    augmented -= augmented[:, PELVIS_IDX : PELVIS_IDX + 1]
    return augmented


def augment_skeleton_sequence(
    coords: np.ndarray,
    *,
    duration_scale: float = 1.0,
    yaw_deg: float = 0.0,
    jitter_std: float = 0.0,
    smoothing: float = 0.80,
    seed: int = 0,
) -> np.ndarray:
    """Apply temporal, yaw, and smooth-jitter transforms deterministically."""
    augmented = temporal_resample(coords, duration_scale)
    augmented = yaw_rotate(augmented, yaw_deg)
    augmented = add_smooth_jitter(
        augmented,
        jitter_std,
        np.random.default_rng(int(seed)),
        smoothing=smoothing,
    )
    return augmented.astype(np.float32)


__all__ = [
    "add_smooth_jitter",
    "augment_skeleton_sequence",
    "temporal_resample",
    "yaw_rotate",
]
