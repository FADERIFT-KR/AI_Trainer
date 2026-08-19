"""스쿼트 phase segmentation을 위한 프레임별 biomechanical feature 추출.

claude.md 9장에서 정한 1차 후보 feature:
    - pelvis normalized height
    - pelvis vertical velocity
    - knee flexion angle
    - hip flexion angle

입력은 ``reference_pipeline``에서 만든 정규화(Hip-center+Scale+Orientation) 완료된
(T, 18, 3) 좌표다. Hip-centered 좌표에서는 Pelvis 자체가 항상 원점이므로, "pelvis
height"는 Ankle이 Pelvis 대비 수직축(orientation alignment의 vertical axis, index 1)
방향으로 얼마나 떨어져 있는지로 정의한다 — 앉을수록(Pelvis가 지면에 가까워질수록)
이 거리가 줄어드는 신호이며, leg_length로 이미 나눠져 있어 "normalized" 조건도 만족한다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_L_HIP, _R_HIP = _IDX["LHip"], _IDX["RHip"]
_L_KNEE, _R_KNEE = _IDX["LKnee"], _IDX["RKnee"]
_L_ANKLE, _R_ANKLE = _IDX["LAnkle"], _IDX["RAnkle"]
_NECK = _IDX["Neck"]
_VERTICAL_AXIS = 1  # orientation_align_3d 결과에서 vertical(y) 축 인덱스


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """b를 꼭짓점으로 하는 벡터 b->a, b->c 사이 각도(도). a,b,c: (...,3)."""
    v1, v2 = a - b, c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


@dataclass
class PhaseFeatures:
    pelvis_height: np.ndarray  # (T,) leg_length 단위로 정규화됨
    pelvis_velocity: np.ndarray  # (T,) pelvis_height의 프레임간 변화량
    knee_flexion_deg: np.ndarray  # (T,) 좌우 평균, 완전 신전=180도에 가까움
    hip_flexion_deg: np.ndarray  # (T,) 좌우 평균 (Neck-Hip-Knee 각)


def extract_phase_features(aligned_coords: np.ndarray) -> PhaseFeatures:
    ankle_vert = (aligned_coords[:, _L_ANKLE, _VERTICAL_AXIS] + aligned_coords[:, _R_ANKLE, _VERTICAL_AXIS]) / 2.0
    pelvis_height = -ankle_vert  # 서 있을수록 큼, 앉을수록 0에 가까워짐

    pelvis_velocity = np.gradient(pelvis_height)

    knee_l = _angle_deg(aligned_coords[:, _L_HIP], aligned_coords[:, _L_KNEE], aligned_coords[:, _L_ANKLE])
    knee_r = _angle_deg(aligned_coords[:, _R_HIP], aligned_coords[:, _R_KNEE], aligned_coords[:, _R_ANKLE])
    knee_flexion = (knee_l + knee_r) / 2.0

    hip_l = _angle_deg(aligned_coords[:, _NECK], aligned_coords[:, _L_HIP], aligned_coords[:, _L_KNEE])
    hip_r = _angle_deg(aligned_coords[:, _NECK], aligned_coords[:, _R_HIP], aligned_coords[:, _R_KNEE])
    hip_flexion = (hip_l + hip_r) / 2.0

    return PhaseFeatures(
        pelvis_height=pelvis_height,
        pelvis_velocity=pelvis_velocity,
        knee_flexion_deg=knee_flexion,
        hip_flexion_deg=hip_flexion,
    )
