"""DTW 비교용 11개 feature 그룹 추출.

입력은 `reference_pipeline`이 만든 정규화(Hip-center+Scale+Orientation) 완료
(T, 18, 3) 좌표다. 모든 feature는 `configs/dtw_feature_weights.json`의
`features` 섹션에 문서화된 shape/단위/normalization을 따른다. 서로 다른
feature가 distance를 지배하지 않도록:
  - 좌표/속도/궤적류: leg_length로 이미 나뉜 정규화 좌표계 안에서 계산 (스케일 O(1))
  - 각도류: 度(degree)를 /180으로 나눠 대략 [0,1] 범위로 맞춤
  - 방향벡터: 단위벡터라 이미 [-1,1] 범위
"""
from __future__ import annotations

import numpy as np

from .common_skeleton import COMMON_JOINT_NAMES

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_L_SHOULDER, _R_SHOULDER = _IDX["LShoulder"], _IDX["RShoulder"]
_L_HIP, _R_HIP = _IDX["LHip"], _IDX["RHip"]
_L_KNEE, _R_KNEE = _IDX["LKnee"], _IDX["RKnee"]
_L_ANKLE, _R_ANKLE = _IDX["LAnkle"], _IDX["RAnkle"]
_L_HEEL, _R_HEEL = _IDX["LHeel"], _IDX["RHeel"]
_L_TOE, _R_TOE = _IDX["LBigToe"], _IDX["RBigToe"]
_NECK, _PELVIS = _IDX["Neck"], _IDX["Hip"]
_LATERAL_AXIS, _VERTICAL_AXIS = 0, 1

FEATURE_NAMES = [
    "joint_coords_3d", "knee_flexion_angle", "hip_flexion_angle", "ankle_angle",
    "torso_inclination", "bone_direction_vectors", "joint_velocity",
    "pelvis_trajectory", "heel_height", "knee_toe_alignment", "left_right_asymmetry",
]


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1, axis=-1), np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, axis=-1) / (n1 * n2 + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def extract_all_features(coords: np.ndarray) -> dict[str, np.ndarray]:
    """coords: (T,18,3) 정규화 완료 좌표 -> {feature_name: (T,dim)}."""
    t = coords.shape[0]

    knee_l = _angle_deg(coords[:, _L_HIP], coords[:, _L_KNEE], coords[:, _L_ANKLE])
    knee_r = _angle_deg(coords[:, _R_HIP], coords[:, _R_KNEE], coords[:, _R_ANKLE])
    hip_l = _angle_deg(coords[:, _NECK], coords[:, _L_HIP], coords[:, _L_KNEE])
    hip_r = _angle_deg(coords[:, _NECK], coords[:, _R_HIP], coords[:, _R_KNEE])
    ankle_l = _angle_deg(coords[:, _L_KNEE], coords[:, _L_ANKLE], coords[:, _L_TOE])
    ankle_r = _angle_deg(coords[:, _R_KNEE], coords[:, _R_ANKLE], coords[:, _R_TOE])

    torso_vec = coords[:, _NECK] - coords[:, _PELVIS]
    up = np.zeros_like(torso_vec)
    up[:, _VERTICAL_AXIS] = 1.0
    torso_inclination = _angle_deg(torso_vec, np.zeros_like(torso_vec), up)  # angle(torso_vec,0,up)

    thigh_dir = _unit((_unit(coords[:, _L_KNEE] - coords[:, _L_HIP]) + _unit(coords[:, _R_KNEE] - coords[:, _R_HIP])) / 2)
    shank_dir = _unit((_unit(coords[:, _L_ANKLE] - coords[:, _L_KNEE]) + _unit(coords[:, _R_ANKLE] - coords[:, _R_KNEE])) / 2)
    torso_dir = _unit(torso_vec)
    bone_dirs = np.concatenate([thigh_dir, shank_dir, torso_dir], axis=-1)  # (T,9)

    joint_coords_flat = coords.reshape(t, -1)  # (T,54)
    velocity = np.vstack([np.zeros((1, joint_coords_flat.shape[1])), np.diff(joint_coords_flat, axis=0)])

    ankle_vert = (coords[:, _L_ANKLE, _VERTICAL_AXIS] + coords[:, _R_ANKLE, _VERTICAL_AXIS]) / 2.0
    pelvis_height = -ankle_vert
    pelvis_velocity = np.gradient(pelvis_height)
    pelvis_trajectory = np.stack([pelvis_height, pelvis_velocity], axis=-1)  # (T,2)

    heel_height = np.stack([coords[:, _L_HEEL, _VERTICAL_AXIS], coords[:, _R_HEEL, _VERTICAL_AXIS]], axis=-1)

    knee_toe_l = coords[:, _L_KNEE, _LATERAL_AXIS] - coords[:, _L_TOE, _LATERAL_AXIS]
    knee_toe_r = coords[:, _R_KNEE, _LATERAL_AXIS] - coords[:, _R_TOE, _LATERAL_AXIS]
    knee_toe_alignment = np.stack([knee_toe_l, knee_toe_r], axis=-1)

    asymmetry = np.stack(
        [
            np.abs(knee_l - knee_r) / 180.0,
            np.abs(hip_l - hip_r) / 180.0,
            np.abs(heel_height[:, 0] - heel_height[:, 1]),
            np.abs(ankle_l - ankle_r) / 180.0,
        ],
        axis=-1,
    )

    return {
        "joint_coords_3d": joint_coords_flat,
        "knee_flexion_angle": np.stack([knee_l, knee_r], axis=-1) / 180.0,
        "hip_flexion_angle": np.stack([hip_l, hip_r], axis=-1) / 180.0,
        "ankle_angle": np.stack([ankle_l, ankle_r], axis=-1) / 180.0,
        "torso_inclination": torso_inclination[:, None] / 180.0,
        "bone_direction_vectors": bone_dirs,
        "joint_velocity": velocity,
        "pelvis_trajectory": pelvis_trajectory,
        "heel_height": heel_height,
        "knee_toe_alignment": knee_toe_alignment,
        "left_right_asymmetry": asymmetry,
    }
