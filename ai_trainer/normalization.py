"""claude.md 6장 정규화 파이프라인 구현: Hip-centered Translation → Body-scale
Normalization → Orientation Alignment (body-centered coordinate system).

이 모듈의 3D 함수들은 Common Skeleton(18노드, ``common_skeleton.py``)을 입력으로 받는다.
Orientation Alignment는 camera1 가설과 무관하게 신체 축(좌우 Hip, Pelvis→Shoulder)만
사용한다.
"""
from __future__ import annotations

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_L_HIP, _R_HIP = _IDX["LHip"], _IDX["RHip"]
_L_KNEE, _R_KNEE = _IDX["LKnee"], _IDX["RKnee"]
_L_ANKLE, _R_ANKLE = _IDX["LAnkle"], _IDX["RAnkle"]
_L_SHOULDER, _R_SHOULDER = _IDX["LShoulder"], _IDX["RShoulder"]
_PELVIS = _IDX["Hip"]


def hip_center_3d(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(..., 18, 3) -> Pelvis(Hip)를 원점으로 이동한 좌표와, 제거된 pelvis 위치를 반환한다.

    프레임마다 독립적으로 중심을 이동한다 (P'_j = P_j - P_pelvis).
    """
    pelvis = coords[..., _PELVIS : _PELVIS + 1, :]
    return coords - pelvis, pelvis[..., 0, :]


def leg_length_scale(centered: np.ndarray) -> np.ndarray:
    """Hip-centered (..., 18, 3) 좌표에서 좌우 다리 길이 평균을 스케일 기준으로 계산한다.

    leg_length = mean( |Hip-Knee|+|Knee-Ankle| (좌), |Hip-Knee|+|Knee-Ankle| (우) ) / 2
    """
    l_len = np.linalg.norm(centered[..., _L_HIP, :] - centered[..., _L_KNEE, :], axis=-1) + np.linalg.norm(
        centered[..., _L_KNEE, :] - centered[..., _L_ANKLE, :], axis=-1
    )
    r_len = np.linalg.norm(centered[..., _R_HIP, :] - centered[..., _R_KNEE, :], axis=-1) + np.linalg.norm(
        centered[..., _R_KNEE, :] - centered[..., _R_ANKLE, :], axis=-1
    )
    return (l_len + r_len) / 2.0


def scale_normalize_3d(centered: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """P''_j = P'_j / leg_length. scale은 (...,) 형태, 프레임마다 나눠준다."""
    return centered / scale[..., None, None]


def body_axes(coords_frame: np.ndarray) -> np.ndarray:
    """한 프레임 (18,3)에서 body-centered 축 [lateral, vertical, forward]을 열벡터로 갖는 3x3 회전행렬을 계산한다.

    - lateral(x)  = normalize(RHip - LHip)
    - vertical(y) = normalize(ShoulderCenter - Pelvis)
    - forward(z)  = normalize(lateral x vertical)
    - vertical을 그람-슈미트로 재직교화: vertical' = normalize(forward x lateral)
    """
    lateral = coords_frame[_R_HIP] - coords_frame[_L_HIP]
    lateral = lateral / (np.linalg.norm(lateral) + 1e-8)

    shoulder_center = (coords_frame[_L_SHOULDER] + coords_frame[_R_SHOULDER]) / 2.0
    vertical = shoulder_center - coords_frame[_PELVIS]
    vertical = vertical / (np.linalg.norm(vertical) + 1e-8)

    forward = np.cross(lateral, vertical)
    forward = forward / (np.linalg.norm(forward) + 1e-8)

    vertical = np.cross(forward, lateral)  # 그람-슈미트 재직교화
    vertical = vertical / (np.linalg.norm(vertical) + 1e-8)

    return np.stack([lateral, vertical, forward], axis=1)  # (3,3), 열이 각 축


def orientation_align_3d(scaled_seq: np.ndarray, reference_frames: int = 5) -> np.ndarray:
    """(T, 18, 3) 시퀀스를 body-centered 좌표계로 정렬한다.

    시퀀스 앞부분(reference_frames, 기본 준비 자세로 가정)의 평균 축을 기준 회전으로
    삼아 **전체 시퀀스에 동일한 회전을 적용**한다. 프레임마다 다시 축을 계산하면
    스쿼트 도중의 의도적인 상체 기울기(고관절오류 판별에 쓰이는 실제 신호)까지
    지워지므로, 외부(카메라/설치) 회전만 제거하고 동작 중 자세 변화는 보존한다.
    """
    n_ref = max(1, min(reference_frames, scaled_seq.shape[0]))
    ref_axes = np.stack([body_axes(scaled_seq[t]) for t in range(n_ref)], axis=0).mean(axis=0)
    # 평균으로 인해 직교성이 깨질 수 있으므로 다시 정규화
    lateral = ref_axes[:, 0] / np.linalg.norm(ref_axes[:, 0])
    forward = np.cross(lateral, ref_axes[:, 1])
    forward /= np.linalg.norm(forward)
    vertical = np.cross(forward, lateral)
    vertical /= np.linalg.norm(vertical)
    R = np.stack([lateral, vertical, forward], axis=1)  # (3,3)

    # world 좌표 -> body-local 좌표: local = R^T @ world
    return np.einsum("ij,tpj->tpi", R.T, scaled_seq)


def normalize_2d_sequence(coords_2d_seq: np.ndarray) -> np.ndarray:
    """(T, 18, 2) 2D 시퀀스를 프레임별 Hip root-center + 시퀀스 공통 스케일로 정규화한다.

    스케일은 프레임마다 다시 계산하지 않고 **시퀀스 전체의 median(Neck-Hip 거리)**
    하나로 고정한다 (스쿼트 중 무릎이 굽혀지며 다리 길이가 원근 축소되는 것과 달리
    몸통 길이는 비교적 안정적이라 스케일 기준으로 사용, 프레임별로 다시 계산하면
    동작 자체가 스케일에 섞여 들어가는 것을 방지).
    """
    pelvis = coords_2d_seq[:, _PELVIS : _PELVIS + 1, :]
    centered = coords_2d_seq - pelvis

    neck_idx = COMMON_JOINT_NAMES.index("Neck")
    torso_len = np.linalg.norm(centered[:, neck_idx, :], axis=-1)  # pelvis 기준이므로 Neck 위치 자체가 거리
    scale = np.median(torso_len)
    scale = max(float(scale), 1e-6)
    return centered / scale
