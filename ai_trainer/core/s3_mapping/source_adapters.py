"""AI Hub 26관절 / MediaPipe 33랜드마크 → Common Skeleton 18노드(미터) 변환.

두 소스를 같은 관절 정의로 맞추는 것이 목적이다. 좌표축 정렬은
``ai_trainer.core.s4_normalize.gravity_frame``에서 한다.

- 단위: AI Hub는 mm, MediaPipe world landmarks는 m → 둘 다 m로 맞춘다.
- Pelvis(``Hip``)·``Neck``: MediaPipe에는 대응 랜드마크가 없어 좌우 고관절·어깨의
  중점으로 만든다. AI Hub가 제공하는 ``Hip``은 고관절 중점보다 약 20mm, ``Neck``은
  어깨 중점보다 약 58mm 위에 있으므로(361개 시퀀스 중앙값), 소스 간 비교에서는
  AI Hub도 같은 중점 정의를 쓴다(``derive_centers=True``).
- 발끝: MediaPipe ``foot_index``(31/32)를 AI Hub ``BigToe``에 대응시킨다. 두 점의
  해부학적 위치가 완전히 같지는 않다.
"""
from __future__ import annotations

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES, to_common_skeleton

# MediaPipe Pose 33 랜드마크 인덱스 (left/right는 피험자 기준)
MEDIAPIPE_INDEX: dict[str, int] = {
    "LShoulder": 11, "RShoulder": 12,
    "LElbow": 13, "RElbow": 14,
    "LWrist": 15, "RWrist": 16,
    "LHip": 23, "RHip": 24,
    "LKnee": 25, "RKnee": 26,
    "LAnkle": 27, "RAnkle": 28,
    "LHeel": 29, "RHeel": 30,
    "LBigToe": 31, "RBigToe": 32,  # foot_index
}

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def _set_centers(common: np.ndarray) -> np.ndarray:
    common[..., _IDX["Hip"], :] = (common[..., _IDX["LHip"], :] + common[..., _IDX["RHip"], :]) / 2.0
    common[..., _IDX["Neck"], :] = (
        common[..., _IDX["LShoulder"], :] + common[..., _IDX["RShoulder"], :]
    ) / 2.0
    return common


def mediapipe_to_common(world_landmarks: np.ndarray) -> np.ndarray:
    """(..., 33, 3) MediaPipe world landmarks(m) → (..., 18, 3) Common Skeleton(m)."""
    if world_landmarks.shape[-2:] != (33, 3):
        raise ValueError(f"expected (..., 33, 3), got {world_landmarks.shape}")
    out = np.zeros(world_landmarks.shape[:-2] + (len(COMMON_JOINT_NAMES), 3), dtype=np.float64)
    for name, mp_index in MEDIAPIPE_INDEX.items():
        out[..., _IDX[name], :] = world_landmarks[..., mp_index, :]
    return _set_centers(out)


def aihub_to_common(coords_26_mm: np.ndarray, derive_centers: bool = True) -> np.ndarray:
    """(..., 26, 3) AI Hub 3D(mm) → (..., 18, 3) Common Skeleton(m).

    ``derive_centers=True``이면 ``Hip``/``Neck``을 MediaPipe와 같은 중점 정의로 바꾼다.
    """
    if coords_26_mm.shape[-2:] != (26, 3):
        raise ValueError(f"expected (..., 26, 3), got {coords_26_mm.shape}")
    out = to_common_skeleton(np.asarray(coords_26_mm, dtype=np.float64)) / 1000.0
    return _set_centers(out) if derive_centers else out
