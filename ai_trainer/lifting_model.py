"""2D->3D Temporal Lifting baseline 모델.

VideoPose3D 계열을 축소한 경량 Dilated Temporal Conv1d + FC head.
입력 (B,T,J,2) 시간창 전체를 보고 **center frame의 3D(Hip-centered)** 하나만 예측한다.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalLiftingNet(nn.Module):
    def __init__(self, n_joints: int = 18, hidden: int = 128, dilations: tuple[int, ...] = (1, 2, 1)):
        super().__init__()
        self.n_joints = n_joints
        in_ch = n_joints * 2
        out_dim = n_joints * 3

        layers = []
        ch = in_ch
        for d in dilations:
            layers += [
                nn.Conv1d(ch, hidden, kernel_size=3, dilation=d, padding=d),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
            ]
            ch = hidden
        self.temporal = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, J, 2)
        b, t, j, c = x.shape
        x = x.reshape(b, t, j * c).permute(0, 2, 1)  # (B, J*2, T)
        feat = self.temporal(x)  # (B, hidden, T)
        center = feat[:, :, t // 2]  # center frame 특징만 사용
        out = self.head(center)  # (B, J*3)
        return out.reshape(b, self.n_joints, 3)
