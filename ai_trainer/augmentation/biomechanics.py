"""Differentiable biomechanical summaries used to guide the generator."""
from __future__ import annotations

import torch

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES

FEATURE_NAMES = (
    "ankle_angle",
    "knee_angle",
    "hip_angle",
    "torso_inclination",
    "heel_lift",
    "pelvis_height",
)

ERROR_FEATURES = {
    "발뒤꿈치오류": ("ankle_angle", "heel_lift"),
    "엉덩이하방오류": ("knee_angle", "hip_angle", "pelvis_height"),
    "고관절오류": ("torso_inclination", "hip_angle"),
}

_IDX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}


def _angle(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    left, right = a - b, c - b
    cosine = (left * right).sum(-1) / (
        torch.linalg.vector_norm(left, dim=-1) * torch.linalg.vector_norm(right, dim=-1) + 1e-7
    )
    return torch.acos(cosine.clamp(-1.0, 1.0)) / torch.pi


def _weighted_mean(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    weight = gate / gate.sum(1, keepdim=True).clamp_min(1e-6)
    return (value * weight).sum(1)


def feature_vector(coords: torch.Tensor, phase_gate: torch.Tensor) -> torch.Tensor:
    """Return six bottom/dynamic-phase features for ``(B,T,18,3)`` input."""
    if coords.ndim != 4 or coords.shape[-2:] != (len(COMMON_JOINT_NAMES), 3):
        raise ValueError("coords must have shape (B,T,18,3)")
    if phase_gate.shape != coords.shape[:2]:
        raise ValueError("phase_gate must have shape (B,T)")

    def bilateral(a: str, b: str, c: str) -> torch.Tensor:
        left = _angle(coords[:, :, _IDX["L" + a]], coords[:, :, _IDX["L" + b]], coords[:, :, _IDX["L" + c]])
        right = _angle(coords[:, :, _IDX["R" + a]], coords[:, :, _IDX["R" + b]], coords[:, :, _IDX["R" + c]])
        return (left + right) / 2.0

    ankle = bilateral("Knee", "Ankle", "BigToe")
    knee = bilateral("Hip", "Knee", "Ankle")
    hip_left = _angle(coords[:, :, _IDX["Neck"]], coords[:, :, _IDX["LHip"]], coords[:, :, _IDX["LKnee"]])
    hip_right = _angle(coords[:, :, _IDX["Neck"]], coords[:, :, _IDX["RHip"]], coords[:, :, _IDX["RKnee"]])
    hip = (hip_left + hip_right) / 2.0

    torso = coords[:, :, _IDX["Neck"]] - coords[:, :, _IDX["Hip"]]
    up = torch.zeros_like(torso)
    up[..., 1] = 1.0
    zero = torch.zeros_like(torso)
    torso_angle = _angle(torso, zero, up)

    heel = (coords[:, :, _IDX["LHeel"], 1] - coords[:, :, _IDX["LBigToe"], 1]
            + coords[:, :, _IDX["RHeel"], 1] - coords[:, :, _IDX["RBigToe"], 1]) / 2.0
    ankle_y = (coords[:, :, _IDX["LAnkle"], 1] + coords[:, :, _IDX["RAnkle"], 1]) / 2.0
    pelvis_height = -ankle_y

    values = [ankle, knee, hip, torso_angle, heel, pelvis_height]
    return torch.stack([_weighted_mean(value, phase_gate) for value in values], dim=-1)
