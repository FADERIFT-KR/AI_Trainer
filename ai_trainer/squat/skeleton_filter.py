"""Robust offline refinement for AI Hub 3-D skeleton trajectories.

The pipeline separates two failure modes which should not be handled by one
blind smoother:

1. short tracking spikes are detected from a rolling median/MAD and replaced
   by temporal interpolation;
2. remaining high-frequency jitter is attenuated by a zero-phase 4th-order
   Butterworth low-pass filter, avoiding temporal lag;
3. major limb lengths are softly restored to their robust sequence medians.

The default 6 Hz cutoff at 30 fps follows common biomechanical marker
processing practice and remains well above the fundamental squat frequency.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, savgol_filter, sosfiltfilt


DEFAULT_BONES = (
    ("Hip", "LHip"), ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("LAnkle", "LHeel"), ("LAnkle", "LBigToe"), ("LAnkle", "LSmallToe"),
    ("Hip", "RHip"), ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("RAnkle", "RHeel"), ("RAnkle", "RBigToe"), ("RAnkle", "RSmallToe"),
    ("Hip", "Neck"), ("Neck", "Head"),
    ("Neck", "LShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("Neck", "RShoulder"), ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
)


@dataclass(frozen=True)
class SkeletonFilterConfig:
    fps: float = 30.0
    cutoff_hz: float = 6.0
    butterworth_order: int = 4
    hampel_window: int = 7
    hampel_sigma: float = 4.5
    minimum_spike_ratio: float = 0.012
    bone_length_strength: float = 0.65

    def __post_init__(self) -> None:
        if self.fps <= 0 or not 0 < self.cutoff_hz < self.fps / 2:
            raise ValueError("cutoff_hz must be between 0 and the Nyquist frequency")
        if self.butterworth_order < 1:
            raise ValueError("butterworth_order must be positive")
        if self.hampel_window < 3 or self.hampel_window % 2 == 0:
            raise ValueError("hampel_window must be an odd integer >= 3")
        if not 0.0 <= self.bone_length_strength <= 1.0:
            raise ValueError("bone_length_strength must be between 0 and 1")


@dataclass(frozen=True)
class SkeletonRefinementReport:
    frames: int
    joints: int
    spike_points: int
    spike_fraction: float
    raw_jerk_rms_normalized: float
    refined_jerk_rms_normalized: float
    raw_bone_length_cv: float
    refined_bone_length_cv: float
    displacement_rms_normalized: float

    def to_dict(self) -> dict:
        return asdict(self)


def _bone_indices(joint_names: Sequence[str]) -> list[tuple[int, int]]:
    index = {name: idx for idx, name in enumerate(joint_names)}
    return [(index[parent], index[child]) for parent, child in DEFAULT_BONES if parent in index and child in index]


def _body_scale(coords: np.ndarray, bones: list[tuple[int, int]]) -> float:
    lengths = [np.linalg.norm(coords[:, child] - coords[:, parent], axis=-1) for parent, child in bones]
    if lengths:
        scale = float(np.median(np.concatenate(lengths)))
    else:
        scale = float(np.median(np.linalg.norm(coords - np.median(coords, axis=1, keepdims=True), axis=-1)))
    return max(scale, 1e-8)


def _interpolate_spikes(coords: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
    output = coords.copy()
    frame_index = np.arange(len(coords))
    for joint in range(coords.shape[1]):
        bad = spike_mask[:, joint]
        if not bad.any():
            continue
        good = ~bad
        if good.sum() < 2:
            continue
        for dim in range(3):
            output[bad, joint, dim] = np.interp(
                frame_index[bad], frame_index[good], coords[good, joint, dim]
            )
    return output


def _despike(
    coords: np.ndarray, config: SkeletonFilterConfig, body_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    window = config.hampel_window
    local_median = median_filter(coords, size=(window, 1, 1), mode="nearest")
    residual = np.linalg.norm(coords - local_median, axis=-1)
    residual_median = median_filter(residual, size=(window, 1), mode="nearest")
    mad = median_filter(np.abs(residual - residual_median), size=(window, 1), mode="nearest")
    robust_sigma = 1.4826 * mad
    threshold = np.maximum(
        residual_median + config.hampel_sigma * robust_sigma,
        config.minimum_spike_ratio * body_scale,
    )
    spikes = residual > threshold
    return _interpolate_spikes(coords, spikes), spikes


def _zero_phase_lowpass(coords: np.ndarray, config: SkeletonFilterConfig) -> np.ndarray:
    if len(coords) < 5:
        return coords.copy()
    sos = butter(
        config.butterworth_order,
        config.cutoff_hz,
        btype="lowpass",
        fs=config.fps,
        output="sos",
    )
    try:
        return sosfiltfilt(sos, coords, axis=0)
    except ValueError:
        window = min(len(coords) if len(coords) % 2 else len(coords) - 1, 9)
        if window < 5:
            return coords.copy()
        return savgol_filter(coords, window_length=window, polyorder=min(3, window - 2), axis=0)


def _stabilize_bone_lengths(
    filtered: np.ndarray,
    robust_source: np.ndarray,
    bones: list[tuple[int, int]],
    strength: float,
) -> np.ndarray:
    if strength <= 0 or not bones:
        return filtered.copy()
    output = filtered.copy()
    targets = {
        (parent, child): float(np.median(np.linalg.norm(robust_source[:, child] - robust_source[:, parent], axis=-1)))
        for parent, child in bones
    }
    for frame in range(len(output)):
        for parent, child in bones:
            vector = filtered[frame, child] - filtered[frame, parent]
            norm = float(np.linalg.norm(vector))
            target = targets[(parent, child)]
            if norm <= 1e-10 or target <= 1e-10:
                continue
            desired = output[frame, parent] + vector / norm * target
            output[frame, child] = (1.0 - strength) * filtered[frame, child] + strength * desired
    return output


def _jerk_rms(coords: np.ndarray, scale: float) -> float:
    if len(coords) < 4:
        return 0.0
    return float(np.sqrt(np.mean(np.diff(coords, n=3, axis=0) ** 2)) / scale)


def _bone_cv(coords: np.ndarray, bones: list[tuple[int, int]]) -> float:
    values = []
    for parent, child in bones:
        lengths = np.linalg.norm(coords[:, child] - coords[:, parent], axis=-1)
        mean = float(np.mean(lengths))
        if mean > 1e-10:
            values.append(float(np.std(lengths) / mean))
    return float(np.mean(values)) if values else 0.0


def refine_skeleton_sequence(
    coords_3d: np.ndarray,
    joint_names: Sequence[str],
    config: SkeletonFilterConfig | None = None,
) -> tuple[np.ndarray, SkeletonRefinementReport]:
    """Refine one ``(T,J,3)`` sequence and return auditable diagnostics."""
    cfg = config or SkeletonFilterConfig()
    coords = np.asarray(coords_3d, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[2] != 3 or coords.shape[1] != len(joint_names):
        raise ValueError(f"coords_3d must have shape (T,{len(joint_names)},3): {coords.shape}")
    if not np.isfinite(coords).all():
        raise ValueError("coords_3d contains NaN or infinite values")
    bones = _bone_indices(joint_names)
    scale = _body_scale(coords, bones)
    despiked, spike_mask = _despike(coords, cfg, scale)
    lowpassed = _zero_phase_lowpass(despiked, cfg)
    refined = _stabilize_bone_lengths(lowpassed, despiked, bones, cfg.bone_length_strength)
    report = SkeletonRefinementReport(
        frames=len(coords),
        joints=coords.shape[1],
        spike_points=int(spike_mask.sum()),
        spike_fraction=float(spike_mask.mean()),
        raw_jerk_rms_normalized=_jerk_rms(coords, scale),
        refined_jerk_rms_normalized=_jerk_rms(refined, scale),
        raw_bone_length_cv=_bone_cv(coords, bones),
        refined_bone_length_cv=_bone_cv(refined, bones),
        displacement_rms_normalized=float(np.sqrt(np.mean((refined - coords) ** 2)) / scale),
    )
    return refined, report


__all__ = [
    "SkeletonFilterConfig",
    "SkeletonRefinementReport",
    "refine_skeleton_sequence",
]
