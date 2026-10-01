from __future__ import annotations

import numpy as np
import torch

from ai_trainer.augmentation.constraints import total_constraint_loss
from ai_trainer.augmentation.model import ConditionalErrorGenerator
from ai_trainer.augmentation.preprocessing import denormalize_sequence, normalize_sequence
from ai_trainer.augmentation.projection import orthographic_project
from ai_trainer.augmentation.schema import ALL_LABELS, SequenceMetadata, SequenceRecord
from ai_trainer.augmentation.training import TrainingConfig, train_all
from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES


def _sequence(frames: int = 83) -> np.ndarray:
    index = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
    base = np.zeros((len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
    positions = {
        "Hip": (0, 1.0, 0), "Neck": (0, 1.7, 0),
        "LHip": (-0.15, 1.0, 0), "RHip": (0.15, 1.0, 0),
        "LKnee": (-0.15, 0.55, 0.05), "RKnee": (0.15, 0.55, 0.05),
        "LAnkle": (-0.15, 0.1, 0), "RAnkle": (0.15, 0.1, 0),
        "LHeel": (-0.15, 0.02, -0.1), "RHeel": (0.15, 0.02, -0.1),
        "LBigToe": (-0.15, 0.02, 0.2), "RBigToe": (0.15, 0.02, 0.2),
        "LShoulder": (-0.25, 1.65, 0), "RShoulder": (0.25, 1.65, 0),
        "LElbow": (-0.4, 1.35, 0), "RElbow": (0.4, 1.35, 0),
        "LWrist": (-0.45, 1.05, 0), "RWrist": (0.45, 1.05, 0),
    }
    for name, value in positions.items():
        base[index[name]] = value
    coords = np.repeat(base[None], frames, axis=0)
    phase = np.sin(np.linspace(0, np.pi, frames))
    coords[:, :, 1] -= 0.25 * phase[:, None]
    coords[:, index["Hip"], 1] += 0.05 * phase
    coords += np.array([1.2, 0.3, -0.7])
    return coords


def test_reversible_normalization() -> None:
    original = _sequence()
    normalized, context = normalize_sequence(original)
    restored = denormalize_sequence(normalized, context)
    old_t = np.linspace(0, 1, len(original))
    new_t = np.linspace(0, 1, 64)
    expected = np.empty_like(restored)
    for joint in range(expected.shape[1]):
        for axis in range(3):
            expected[:, joint, axis] = np.interp(new_t, old_t, original[:, joint, axis])
    np.testing.assert_allclose(restored, expected, atol=2e-6)


def test_generator_preserves_frames_outside_gate() -> None:
    normalized, _ = normalize_sequence(_sequence())
    source = torch.from_numpy(normalized[None])
    gate = torch.zeros(1, 64)
    model = ConditionalErrorGenerator(width=16)
    generated = model(source, torch.tensor([0]), torch.ones(1, 1), gate)
    torch.testing.assert_close(generated, source)


def test_identity_has_zero_constraint_loss() -> None:
    normalized, _ = normalize_sequence(_sequence())
    source = torch.from_numpy(normalized[None])
    total, parts = total_constraint_loss(source, source, torch.ones(1, 64))
    assert float(total) == 0.0
    assert all(float(value) == 0.0 for value in parts.values())


def test_projection_shape() -> None:
    normalized, _ = normalize_sequence(_sequence())
    projected = orthographic_project(normalized, yaw_degrees=30)
    assert projected.shape == (64, 18, 2)
    assert np.isfinite(projected).all()


def test_one_epoch_training_smoke(tmp_path) -> None:
    normalized, _ = normalize_sequence(_sequence())
    gate = np.ones(64, dtype=np.float32)
    records = []
    for index, label in enumerate(ALL_LABELS):
        coords = normalized.copy()
        coords[:, :, 2] += index * 0.01 * np.sin(np.linspace(0, np.pi, 64))[:, None]
        records.append(
            SequenceRecord(
                coords,
                label,
                gate,
                SequenceMetadata("test", f"actor{index}", "test", "1", label, "train"),
            )
        )
    output = tmp_path / "smoke.pt"
    metrics = train_all(
        records,
        records,
        output,
        TrainingConfig(batch_size=4, classifier_epochs=1, generator_epochs=1),
        "cpu",
    )
    assert output.exists()
    assert output.with_suffix(".json").exists()
    assert set(metrics) == {
        "classifier_train_accuracy",
        "classifier_validation_accuracy",
        "generated_validation_acceptance_rate",
    }
