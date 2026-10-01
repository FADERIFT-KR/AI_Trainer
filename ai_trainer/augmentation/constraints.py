"""Structural losses that keep generated motion anatomically plausible."""
from __future__ import annotations

import torch

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_BONE_INDEX_PAIRS, COMMON_JOINT_NAMES

_IDX = {name: index for index, name in enumerate(COMMON_JOINT_NAMES)}
_FOOT = [_IDX[name] for name in ("LBigToe", "RBigToe")]


def bone_length_loss(generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    pairs = torch.as_tensor(COMMON_BONE_INDEX_PAIRS, device=generated.device)
    generated_length = torch.linalg.vector_norm(
        generated[:, :, pairs[:, 0]] - generated[:, :, pairs[:, 1]], dim=-1
    )
    source_length = torch.linalg.vector_norm(
        source[:, :, pairs[:, 0]] - source[:, :, pairs[:, 1]], dim=-1
    )
    relative = (generated_length - source_length) / source_length.clamp_min(1e-4)
    return relative.square().mean()


def temporal_smoothness_loss(generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    delta = generated - source
    if delta.shape[1] < 3:
        return delta.new_zeros(())
    acceleration = delta[:, 2:] - 2.0 * delta[:, 1:-1] + delta[:, :-2]
    return acceleration.square().mean()


def foot_anchor_loss(generated: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    return (generated[:, :, _FOOT] - source[:, :, _FOOT]).square().mean()


def outside_phase_loss(
    generated: torch.Tensor, source: torch.Tensor, phase_gate: torch.Tensor
) -> torch.Tensor:
    outside = (1.0 - phase_gate)[:, :, None, None]
    return (((generated - source) * outside) ** 2).mean()


def total_constraint_loss(
    generated: torch.Tensor, source: torch.Tensor, phase_gate: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts = {
        "bone": bone_length_loss(generated, source),
        "smooth": temporal_smoothness_loss(generated, source),
        "foot": foot_anchor_loss(generated, source),
        "outside": outside_phase_loss(generated, source, phase_gate),
    }
    total = 5.0 * parts["bone"] + parts["smooth"] + 2.0 * parts["foot"] + 5.0 * parts["outside"]
    return total, parts
