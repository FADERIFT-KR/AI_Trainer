"""Independent structural and semantic checks for generated samples."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_BONE_INDEX_PAIRS, COMMON_JOINT_NAMES

from .biomechanics import ERROR_FEATURES, FEATURE_NAMES, feature_vector
from .rules import RuleProfile


@dataclass(frozen=True)
class ValidationReport:
    accepted: bool
    finite: bool
    max_bone_relative_error: float
    acceleration_rms: float
    toe_velocity_rms: float
    target_profile_distance: float
    normal_profile_distance: float
    target_closer_than_normal: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _bone_error(generated: np.ndarray, source: np.ndarray) -> float:
    pairs = np.asarray(COMMON_BONE_INDEX_PAIRS)
    generated_length = np.linalg.norm(generated[:, pairs[:, 0]] - generated[:, pairs[:, 1]], axis=-1)
    source_length = np.linalg.norm(source[:, pairs[:, 0]] - source[:, pairs[:, 1]], axis=-1)
    relative = np.abs(generated_length - source_length) / np.maximum(source_length, 1e-6)
    return float(np.max(relative))


def validate_generated(
    generated: np.ndarray,
    source: np.ndarray,
    phase_gate: np.ndarray,
    target_label: str,
    profile: RuleProfile,
) -> ValidationReport:
    finite = bool(np.isfinite(generated).all())
    if not finite:
        return ValidationReport(
            False, False, float("inf"), float("inf"), float("inf"),
            float("inf"), float("inf"), False,
        )
    with torch.no_grad():
        values = feature_vector(
            torch.as_tensor(generated[None], dtype=torch.float32),
            torch.as_tensor(phase_gate[None], dtype=torch.float32),
        )[0].numpy()
    selected = np.array([name in ERROR_FEATURES[target_label] for name in FEATURE_NAMES])
    target_z = np.abs((values - profile.means[target_label]) / profile.scales[target_label])[selected]
    normal_z = np.abs((values - profile.means["정상"]) / profile.scales["정상"])[selected]
    target_distance = float(target_z.mean())
    normal_distance = float(normal_z.mean())
    bone_error = _bone_error(generated, source)
    acceleration_rms = float(np.sqrt(np.mean(np.diff(generated, n=2, axis=0) ** 2)))
    toe_indices = [COMMON_JOINT_NAMES.index("LBigToe"), COMMON_JOINT_NAMES.index("RBigToe")]
    toe_velocity_rms = float(np.sqrt(np.mean(np.diff(generated[:, toe_indices], axis=0) ** 2)))
    closer = target_distance < normal_distance
    accepted = (
        bone_error <= profile.max_bone_relative_error
        and acceleration_rms <= profile.max_acceleration_rms
        and toe_velocity_rms <= profile.max_toe_velocity_rms
        and target_distance <= profile.max_target_z
        and closer
    )
    return ValidationReport(
        accepted,
        finite,
        bone_error,
        acceleration_rms,
        toe_velocity_rms,
        target_distance,
        normal_distance,
        closer,
    )
