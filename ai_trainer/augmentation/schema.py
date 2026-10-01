"""Shared types and labels for generated squat skeleton datasets."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

NORMAL_LABEL = "정상"
ERROR_LABELS = ("발뒤꿈치오류", "엉덩이하방오류", "고관절오류")
ALL_LABELS = (NORMAL_LABEL, *ERROR_LABELS)
LABEL_TO_ID = {label: index for index, label in enumerate(ALL_LABELS)}
ERROR_TO_ID = {label: index for index, label in enumerate(ERROR_LABELS)}
N_FRAMES = 64
N_JOINTS = 18


@dataclass(frozen=True)
class SequenceMetadata:
    source: str
    actor: str
    level: str
    repetition: str
    original_label: str
    split: str
    camera: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NormalizationContext:
    pelvis: np.ndarray
    scale: float
    rotation: np.ndarray


@dataclass
class SequenceRecord:
    """One normalized, fixed-length 3-D squat sequence."""

    coords: np.ndarray
    label: str
    phase_gate: np.ndarray
    metadata: SequenceMetadata
    context: NormalizationContext | None = None

    def __post_init__(self) -> None:
        self.coords = validate_skeleton(self.coords)
        gate = np.asarray(self.phase_gate, dtype=np.float32)
        if gate.shape != (N_FRAMES,) or not np.isfinite(gate).all():
            raise ValueError(f"phase_gate must be finite ({N_FRAMES},), got {gate.shape}")
        self.phase_gate = np.clip(gate, 0.0, 1.0)
        if self.label not in ALL_LABELS:
            raise ValueError(f"unsupported label: {self.label}")


def validate_skeleton(coords: np.ndarray) -> np.ndarray:
    value = np.asarray(coords, dtype=np.float32)
    expected = (N_FRAMES, N_JOINTS, 3)
    if value.shape != expected:
        raise ValueError(f"skeleton must have shape {expected}, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("skeleton contains NaN or infinite values")
    return value
