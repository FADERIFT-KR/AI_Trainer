"""Small multi-task spatial-temporal graph network for squat error diagnosis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .common_skeleton import COMMON_BONE_INDEX_PAIRS, COMMON_JOINT_NAMES
from .two_stage_squat import ERROR_CLASSES, PARTS, T_REF, mask_occluded_joints


def adjacency_matrix() -> torch.Tensor:
    n = len(COMMON_JOINT_NAMES)
    matrix = torch.eye(n, dtype=torch.float32)
    for a, b in COMMON_BONE_INDEX_PAIRS:
        matrix[a, b] = matrix[b, a] = 1.0
    degree = matrix.sum(1).pow(-0.5)
    return degree[:, None] * matrix * degree[None, :]


class STGCNBlock(nn.Module):
    """Bone-neighbor aggregation followed by temporal convolution."""

    def __init__(self, channels_in: int, channels_out: int, adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("adjacency", adjacency)
        self.spatial = nn.Conv2d(channels_in, channels_out, kernel_size=1)
        self.temporal = nn.Conv2d(channels_out, channels_out, kernel_size=(5, 1), padding=(2, 0))
        self.norm = nn.BatchNorm2d(channels_out)
        self.shortcut = nn.Identity() if channels_in == channels_out else nn.Conv2d(channels_in, channels_out, 1)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial = torch.einsum("bctv,vw->bctw", x, self.adjacency)
        return self.activation(self.norm(self.temporal(self.spatial(spatial))) + self.shortcut(x))


class MultiTaskSTGCN(nn.Module):
    """Shared graph feature extractor with error and multilabel body-part heads."""

    def __init__(self, width: int = 32):
        super().__init__()
        adjacency = adjacency_matrix()
        self.blocks = nn.Sequential(
            STGCNBlock(3, width, adjacency),
            STGCNBlock(width, width, adjacency),
            STGCNBlock(width, width * 2, adjacency),
        )
        self.dropout = nn.Dropout(0.25)
        self.error_head = nn.Linear(width * 2, len(ERROR_CLASSES))
        self.part_head = nn.Linear(width * 2, len(PARTS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1:] != (3, T_REF, len(COMMON_JOINT_NAMES)):
            raise ValueError("MT-ST-GCN input must have shape (batch, 3, 64, 18)")
        shared = self.dropout(self.blocks(x).mean(dim=(2, 3)))
        return self.error_head(shared), self.part_head(shared)


class SquatErrorDiagnoser:
    def __init__(self, model: MultiTaskSTGCN):
        self.model = model.eval()

    @classmethod
    def load(cls, path: str | Path) -> "SquatErrorDiagnoser":
        model = MultiTaskSTGCN()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return cls(model)

    def predict(self, aligned_track: np.ndarray, view: str = "front") -> dict:
        x = np.asarray(aligned_track, dtype=np.float32)
        if x.shape != (T_REF, len(COMMON_JOINT_NAMES), 3) or not np.isfinite(x).all():
            raise ValueError("Expected finite aligned (64,18,3) skeleton")
        x = mask_occluded_joints(x, view)
        with torch.no_grad():
            error_logits, part_logits = self.model(torch.from_numpy(x.transpose(2, 0, 1)[None]))
            error_prob = torch.softmax(error_logits[0], 0).numpy()
            part_prob = torch.sigmoid(part_logits[0]).numpy()
        return {
            "error": ERROR_CLASSES[int(error_prob.argmax())],
            "error_probabilities": dict(zip(ERROR_CLASSES, map(float, error_prob))),
            "body_part_probabilities": dict(zip(PARTS, map(float, part_prob))),
        }
