"""Long-context monocular 2D-to-3D pose lifting for depth-sensitive checks.

This implementation follows the residual dilated temporal-convolution design of
VideoPose3D (Pavllo et al., CVPR 2019), with a 27-frame symmetric receptive
field.  It predicts only the centre frame.  The symmetric context is intentional:
it gives a more stable offline/replay estimate but adds 13-frame latency, so it is
not silently substituted for MediaPipe's live world landmarks.
"""
from __future__ import annotations

import torch
from torch import nn

from ai_trainer.core.common_skeleton import COMMON_BONE_INDEX_PAIRS


TEMPORAL_WINDOW = 27


class _ResidualDilatedBlock(nn.Module):
    """A valid dilated 3x1 temporal convolution plus the aligned residual."""

    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.dilation = dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.bn = nn.BatchNorm1d(channels)
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn_pointwise = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Valid convolution shrinks the time axis by 2*dilation.  Crop the
        # residual identically so a 27-frame input ends at one centre frame.
        residual = x[:, :, self.dilation : -self.dilation]
        x = self.dropout(self.relu(self.bn(self.conv(x))))
        x = self.dropout(self.relu(self.bn_pointwise(self.pointwise(x))))
        return x + residual


class VideoPose3DDepthNet(nn.Module):
    """27-frame residual dilated TCN that returns hip-centred 3D in millimetres."""

    temporal_window = TEMPORAL_WINDOW

    def __init__(self, n_joints: int = 18, channels: int = 128, dropout: float = 0.15) -> None:
        super().__init__()
        self.n_joints = n_joints
        self.input = nn.Conv1d(n_joints * 2, channels, kernel_size=3)
        self.input_bn = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        # For filter widths 3,3,3, the dilations are 3 and 9.  Receptive field:
        # 3 + 2*3 + 2*9 = 27 frames.
        self.blocks = nn.ModuleList([
            _ResidualDilatedBlock(channels, dilation=3, dropout=dropout),
            _ResidualDilatedBlock(channels, dilation=9, dropout=dropout),
        ])
        self.output = nn.Conv1d(channels, n_joints * 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (self.temporal_window, self.n_joints, 2):
            raise ValueError(
                "Expected (B, 27, n_joints, 2) normalized 2D input, got "
                f"{tuple(x.shape)}"
            )
        batch = x.shape[0]
        x = x.reshape(batch, self.temporal_window, self.n_joints * 2).transpose(1, 2)
        x = self.dropout(self.relu(self.input_bn(self.input(x))))
        for block in self.blocks:
            x = block(x)
        return self.output(x).squeeze(-1).reshape(batch, self.n_joints, 3)


def depth_aware_lifting_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    z_weight: float = 2.0,
    bone_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Coordinate loss with explicit depth and bone-length supervision.

    The target comes from the dataset's calibrated multi-camera 3D.  Doubling
    the z term prevents the much easier image-plane axes from dominating model
    selection.  Bone lengths are not imposed at inference; they are a training
    regularizer, avoiding a false assumption that every user's body is identical.
    """
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("prediction and target must have the same (B, J, 3) shape")
    axis_mse = (prediction - target).square().mean(dim=(0, 1))
    coordinate = axis_mse[0] + axis_mse[1] + z_weight * axis_mse[2]
    pairs = torch.as_tensor(COMMON_BONE_INDEX_PAIRS, device=prediction.device)
    pred_len = torch.linalg.vector_norm(prediction[:, pairs[:, 0]] - prediction[:, pairs[:, 1]], dim=-1)
    true_len = torch.linalg.vector_norm(target[:, pairs[:, 0]] - target[:, pairs[:, 1]], dim=-1)
    bone = (pred_len - true_len).square().mean()
    total = coordinate + bone_weight * bone
    return total, {"x_mse": axis_mse[0], "y_mse": axis_mse[1], "z_mse": axis_mse[2], "bone_mse": bone}


__all__ = ["TEMPORAL_WINDOW", "VideoPose3DDepthNet", "depth_aware_lifting_loss"]
