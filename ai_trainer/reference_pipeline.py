"""Ground Truth Reference / Operational Reference 시퀀스 생성.

두 계층 모두 동일한 전처리를 거친다:
    Common Skeleton Mapping -> Hip-centered Translation -> Scale Normalization
    -> Body-centered Orientation Alignment

- Ground Truth Reference: AI Hub `3d_points.csv`를 직접 사용.
- Operational Reference : camera1 2D를 학습된 Temporal Lifting 모델에 통과시켜 만든 3D.
  (모델은 이미 Hip-centered 3D를 예측하도록 학습되었으므로, 모델 출력에는
  Scale Normalization과 Orientation Alignment만 이어서 적용한다.)

두 계층 모두 annotation.json의 ``start_frame~end_frame`` 구간만 사용한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .aihub_zip import AiHubZip, SequenceKey
from .common_skeleton import to_common_skeleton
from .lifting_dataset import CAMERA, WINDOW_T
from .lifting_model import TemporalLiftingNet
from .normalization import (
    hip_center_3d,
    leg_length_scale,
    normalize_2d_sequence,
    orientation_align_3d,
    scale_normalize_3d,
)


@dataclass
class ReferenceSequence:
    coords: np.ndarray  # (T, 18, 3) 정규화(Hip-center+Scale+Orientation) 완료된 좌표
    tier: str  # "ground_truth" or "operational"
    seq: SequenceKey
    origin: str
    frame_range: tuple[int, int]  # (start_frame, end_frame), 원본 클립 기준
    scale: float  # leg_length (정규화에 사용한 스케일 값)


def _finalize(hip_centered_seq: np.ndarray) -> tuple[np.ndarray, float]:
    """Hip-centered (T,18,3) -> Scale Normalize -> Orientation Align."""
    scale_per_frame = leg_length_scale(hip_centered_seq)
    scale = float(np.median(scale_per_frame))
    scale = max(scale, 1e-6)
    scaled = hip_centered_seq / scale
    aligned = orientation_align_3d(scaled, reference_frames=5)
    return aligned, scale


def build_ground_truth_reference(z: AiHubZip, seq: SequenceKey, origin: str) -> ReferenceSequence | None:
    frames_3d, coords_26 = z.read_3d(seq)
    ann = z.read_annotation(seq, CAMERA if CAMERA in z.list_cameras(seq) else z.list_cameras(seq)[0])

    n = len(frames_3d)
    start_f, end_f = 0, n - 1
    if ann.get("annotations"):
        a0 = ann["annotations"][0]
        start_f = max(0, min(a0["start_frame"], n - 1))
        end_f = max(start_f, min(a0["end_frame"], n - 1))
    if end_f - start_f < 3:
        return None

    coords_18 = to_common_skeleton(coords_26[start_f : end_f + 1])  # (T,18,3)
    centered, _ = hip_center_3d(coords_18)
    aligned, scale = _finalize(centered)

    return ReferenceSequence(
        coords=aligned.astype(np.float32),
        tier="ground_truth",
        seq=seq,
        origin=origin,
        frame_range=(start_f, end_f),
        scale=scale,
    )


def build_operational_reference(
    z: AiHubZip, seq: SequenceKey, origin: str, model: TemporalLiftingNet, device: torch.device
) -> ReferenceSequence | None:
    if CAMERA not in z.list_cameras(seq):
        return None
    frames_2d, coords_26 = z.read_2d(seq, CAMERA)
    ann = z.read_annotation(seq, CAMERA)

    n = len(frames_2d)
    start_f, end_f = 0, n - 1
    if ann.get("annotations"):
        a0 = ann["annotations"][0]
        start_f = max(0, min(a0["start_frame"], n - 1))
        end_f = max(start_f, min(a0["end_frame"], n - 1))
    if end_f - start_f < 3:
        return None

    coords_18 = to_common_skeleton(coords_26)  # (n,18,2), 전체 클립 (윈도우 문맥용)
    norm_2d_full = normalize_2d_sequence(coords_18)

    half = WINDOW_T // 2
    preds = []
    model.eval()
    with torch.no_grad():
        for c in range(start_f, end_f + 1):
            idxs = np.clip(np.arange(c - half, c + half + 1), 0, n - 1)
            window = norm_2d_full[idxs][None]  # (1,T,18,2)
            x = torch.from_numpy(window.astype(np.float32)).to(device)
            pred = model(x)[0].cpu().numpy()  # (18,3), 이미 Hip-centered (모델 학습 타깃과 동일)
            preds.append(pred)
    hip_centered_seq = np.stack(preds, axis=0)  # (T,18,3)

    aligned, scale = _finalize(hip_centered_seq)

    return ReferenceSequence(
        coords=aligned.astype(np.float32),
        tier="operational",
        seq=seq,
        origin=origin,
        frame_range=(start_f, end_f),
        scale=scale,
    )
