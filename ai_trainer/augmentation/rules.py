"""Data-derived error profiles; semantic feature membership is the only fixed rule."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_BONE_INDEX_PAIRS, COMMON_JOINT_NAMES

from .biomechanics import ERROR_FEATURES, FEATURE_NAMES, feature_vector
from .schema import ALL_LABELS, ERROR_LABELS, SequenceRecord


@dataclass(frozen=True)
class RuleProfile:
    means: dict[str, np.ndarray]
    scales: dict[str, np.ndarray]
    max_bone_relative_error: float
    max_acceleration_rms: float
    max_toe_velocity_rms: float
    max_target_z: float = 3.0

    @classmethod
    def fit(cls, records: list[SequenceRecord]) -> "RuleProfile":
        grouped: dict[str, list[np.ndarray]] = {label: [] for label in ALL_LABELS}
        for record in records:
            coords = torch.from_numpy(record.coords[None])
            gate = torch.from_numpy(record.phase_gate[None])
            grouped[record.label].append(feature_vector(coords, gate)[0].numpy())
        missing = [label for label, values in grouped.items() if not values]
        if missing:
            raise ValueError(f"cannot fit rule profile; missing labels: {missing}")
        means, scales = {}, {}
        for label, values in grouped.items():
            matrix = np.stack(values)
            means[label] = matrix.mean(0).astype(np.float32)
            scales[label] = np.maximum(matrix.std(0), 0.02).astype(np.float32)
        pairs = np.asarray(COMMON_BONE_INDEX_PAIRS)
        bone_deviations = []
        acceleration_rms = []
        toe_velocity_rms = []
        toe_indices = [COMMON_JOINT_NAMES.index("LBigToe"), COMMON_JOINT_NAMES.index("RBigToe")]
        for record in records:
            lengths = np.linalg.norm(
                record.coords[:, pairs[:, 0]] - record.coords[:, pairs[:, 1]], axis=-1
            )
            median = np.median(lengths, axis=0, keepdims=True)
            bone_deviations.append(np.abs(lengths - median) / np.maximum(median, 1e-6))
            acceleration_rms.append(float(np.sqrt(np.mean(np.diff(record.coords, n=2, axis=0) ** 2))))
            toe_velocity_rms.append(
                float(np.sqrt(np.mean(np.diff(record.coords[:, toe_indices], axis=0) ** 2)))
            )
        learned_bone_limit = float(np.quantile(np.concatenate(bone_deviations), 0.99))
        return cls(
            means,
            scales,
            max(0.02, learned_bone_limit),
            max(1e-4, float(np.quantile(acceleration_rms, 0.99))),
            max(1e-4, float(np.quantile(toe_velocity_rms, 0.99))),
        )

    def tensors(self, labels: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(np.stack([self.means[label] for label in labels]), device=device)
        scale = torch.as_tensor(np.stack([self.scales[label] for label in labels]), device=device)
        masks = []
        for label in labels:
            selected = set(ERROR_FEATURES[label])
            masks.append([1.0 if name in selected else 0.0 for name in FEATURE_NAMES])
        mask = torch.tensor(masks, dtype=mean.dtype, device=device)
        return mean, scale, mask

    def to_dict(self) -> dict:
        return {
            "feature_names": list(FEATURE_NAMES),
            "error_features": {key: list(value) for key, value in ERROR_FEATURES.items()},
            "means": {key: value.tolist() for key, value in self.means.items()},
            "scales": {key: value.tolist() for key, value in self.scales.items()},
            "max_bone_relative_error": self.max_bone_relative_error,
            "max_acceleration_rms": self.max_acceleration_rms,
            "max_toe_velocity_rms": self.max_toe_velocity_rms,
            "max_target_z": self.max_target_z,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RuleProfile":
        if tuple(payload.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("rule profile feature schema changed; retrain the checkpoint")
        return cls(
            {key: np.asarray(value, dtype=np.float32) for key, value in payload["means"].items()},
            {key: np.asarray(value, dtype=np.float32) for key, value in payload["scales"].items()},
            float(payload["max_bone_relative_error"]),
            float(payload["max_acceleration_rms"]),
            float(payload["max_toe_velocity_rms"]),
            float(payload.get("max_target_z", 3.0)),
        )


def feature_profile_loss(
    generated: torch.Tensor,
    phase_gate: torch.Tensor,
    target_labels: list[str],
    profile: RuleProfile,
) -> torch.Tensor:
    values = feature_vector(generated, phase_gate)
    mean, scale, mask = profile.tensors(target_labels, generated.device)
    return ((((values - mean) / scale) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


__all__ = ["RuleProfile", "feature_profile_loss", "ERROR_LABELS"]
