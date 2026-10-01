"""Conditional residual network for counterfactual squat-error generation."""
from __future__ import annotations

import torch
from torch import nn

from .schema import ALL_LABELS, ERROR_LABELS, N_FRAMES, N_JOINTS


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.net(value)


class ConditionalErrorGenerator(nn.Module):
    """Generate a gated residual while retaining the source person's motion."""

    def __init__(self, width: int = 128, condition_size: int = 16, noise_size: int = 8) -> None:
        super().__init__()
        self.condition_size = condition_size
        self.noise_size = noise_size
        self.embedding = nn.Embedding(len(ERROR_LABELS), condition_size)
        channels_in = N_JOINTS * 3 + condition_size + noise_size + 1
        self.network = nn.Sequential(
            nn.Conv1d(channels_in, width, 5, padding=2),
            nn.GELU(),
            _ResidualBlock(width, 1),
            _ResidualBlock(width, 2),
            _ResidualBlock(width, 4),
            nn.Conv1d(width, N_JOINTS * 3, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        normal: torch.Tensor,
        error_id: torch.Tensor,
        strength: torch.Tensor,
        phase_gate: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if normal.ndim != 4 or normal.shape[1:] != (N_FRAMES, N_JOINTS, 3):
            raise ValueError("normal must have shape (B,64,18,3)")
        batch = normal.shape[0]
        if error_id.shape != (batch,) or phase_gate.shape != (batch, N_FRAMES):
            raise ValueError("condition shapes do not match the input batch")
        strength = strength.reshape(batch, 1).clamp(0.0, 1.5)
        if noise is None:
            noise = torch.randn(batch, self.noise_size, device=normal.device, dtype=normal.dtype)
        condition = self.embedding(error_id).unsqueeze(-1).expand(-1, -1, N_FRAMES)
        noise_channel = noise.unsqueeze(-1).expand(-1, -1, N_FRAMES)
        strength_channel = strength.unsqueeze(-1).expand(-1, 1, N_FRAMES)
        flat = normal.reshape(batch, N_FRAMES, N_JOINTS * 3).transpose(1, 2)
        delta = self.network(torch.cat([flat, condition, noise_channel, strength_channel], dim=1))
        delta = delta.transpose(1, 2).reshape_as(normal)
        pelvis = 16
        delta = delta - delta[:, :, pelvis : pelvis + 1]
        gate = phase_gate[:, :, None, None]
        amplitude = strength.reshape(batch, 1, 1, 1)
        return normal + 0.25 * amplitude * gate * delta


class ErrorClassifier(nn.Module):
    """Small verifier trained only on real AI Hub sequences."""

    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(N_JOINTS * 3, width, 5, padding=2), nn.GELU(),
            _ResidualBlock(width, 1), _ResidualBlock(width, 2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Linear(width, len(ALL_LABELS))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        flat = coords.reshape(len(coords), N_FRAMES, N_JOINTS * 3).transpose(1, 2)
        return self.output(self.network(flat).squeeze(-1))
