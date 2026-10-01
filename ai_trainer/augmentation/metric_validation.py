"""Compare estimated 3-D squat joints with independently labelled 3-D joints.

This is the coordinate-only part of the project's existing metric-check flow.
Both inputs must be frame-aligned, in metres, and share one fixed axis convention.
Per-frame Procrustes alignment is supplementary: it can hide depth mistakes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES

_HIP = COMMON_JOINT_NAMES.index("Hip")


@dataclass(frozen=True)
class MetricReport:
    frames: int
    valid_frames: int
    root_mpjpe_mm: float
    pa_mpjpe_mm: float
    z_mae_mm: float
    pck_percent: float
    pck_threshold_mm: float
    per_joint_mm: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def _procrustes_align(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align one pose's translation, rotation and scale, without reflection."""
    pred = predicted - predicted.mean(axis=0, keepdims=True)
    ref = target - target.mean(axis=0, keepdims=True)
    u, singular_values, vt = np.linalg.svd(pred.T @ ref)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float((singular_values * np.diag(correction)).sum()) / max(float((pred**2).sum()), 1e-12)
    return pred @ rotation * scale + target.mean(axis=0, keepdims=True)


def compare_sequences(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    *,
    pck_threshold_mm: float = 100.0,
) -> MetricReport:
    """Evaluate corresponding ``(T,18,3)`` sequences without video files.

    Input coordinates must be in metres. The primary root MPJPE and Z MAE
    require the same X/Y/Z axis convention in both sequences. PA MPJPE is
    reported only as a secondary, alignment-tolerant diagnostic.
    """
    pred = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(ground_truth, dtype=np.float64)
    if pred.ndim != 3 or pred.shape[1:] != (len(COMMON_JOINT_NAMES), 3):
        raise ValueError("predicted must have shape (T,18,3)")
    if truth.shape != pred.shape or len(pred) == 0:
        raise ValueError("ground_truth must have the same nonempty (T,18,3) shape")
    if not np.isfinite(truth).all():
        raise ValueError("ground_truth contains non-finite coordinates")
    if pck_threshold_mm <= 0:
        raise ValueError("pck_threshold_mm must be positive")

    valid_frames = np.isfinite(pred).all(axis=(1, 2))
    if not valid_frames.any():
        raise ValueError("no valid predicted frames")
    pred = pred[valid_frames]
    truth = truth[valid_frames]
    pred_root = pred - pred[:, _HIP : _HIP + 1]
    truth_root = truth - truth[:, _HIP : _HIP + 1]

    error_mm = np.linalg.norm(pred_root - truth_root, axis=-1) * 1000.0
    aligned = np.stack([_procrustes_align(p, t) for p, t in zip(pred_root, truth_root)])
    aligned_error_mm = np.linalg.norm(aligned - truth_root, axis=-1) * 1000.0
    return MetricReport(
        frames=len(predicted),
        valid_frames=int(valid_frames.sum()),
        root_mpjpe_mm=float(error_mm.mean()),
        pa_mpjpe_mm=float(aligned_error_mm.mean()),
        z_mae_mm=float(np.abs(pred_root[..., 2] - truth_root[..., 2]).mean() * 1000.0),
        pck_percent=float((error_mm <= pck_threshold_mm).mean() * 100.0),
        pck_threshold_mm=float(pck_threshold_mm),
        per_joint_mm={
            name: float(error_mm[:, joint].mean())
            for joint, name in enumerate(COMMON_JOINT_NAMES)
        },
    )
