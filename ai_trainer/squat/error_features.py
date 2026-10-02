"""스쿼트 반복 1회의 오류 판정 지표.

입력은 ``gravity_frame.align_sequence``로 정렬한 (T, 18, 3) 좌표다
(x = 오른쪽, y = 정면, z = 위, 프레임별 Pelvis 원점). AI Hub와 MediaPipe가 같은
함수를 거치도록 한 프레임 안의 상대 위치·각도만 쓴다. 좌우 값은 평균한다.

각도 부호:
    - 기울기(``*_incl``): 수직축과 이루는 각, 0 = 수직
    - 고도(``*_elev``): 수평면 위쪽이 +
"""
from __future__ import annotations

import numpy as np

from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.core.s4_normalize.gravity_frame import joint_angle_deg

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_SIDES = ("L", "R")

FEATURE_NAMES: tuple[str, ...] = (
    "foot_angle_rise",  # 발 각도(발끝→뒤꿈치 고도)의 반복 중 최댓값 - 선 자세 값
    "heel_rise_norm",  # (뒤꿈치 - 발끝 높이)의 최댓값 - 선 자세 값, 다리 길이 단위
    "knee_min",  # 무릎 내각(발목-무릎-고관절) 최솟값
    "hip_min",  # 고관절 내각(무릎-고관절-어깨) 최솟값
    "ankle_min",  # 발목 내각(무릎-발목-발끝) 최솟값
    "thigh_elev_bottom",  # 최저점 무릎→고관절 고도, +면 고관절이 무릎보다 위
    "tibia_incl_bottom",  # 최저점 정강이 기울기
    "trunk_incl_bottom",  # 최저점 몸통(Pelvis→어깨 중점) 기울기
    "trunk_minus_tibia",  # 최저점 몸통 기울기 - 정강이 기울기
    "trunk_minus_tibia_deep",  # 하강 깊이 절반 아래 구간 평균 (몸통 - 정강이)
    "hip_knee_flex_ratio",  # 최저점 (180-고관절각)/(180-무릎각): 고관절 굴곡 / 무릎 굴곡
    "hip_back_norm",  # 최저점 Pelvis가 발목 중점보다 뒤에 있는 거리, 다리 길이 단위
    "knee_ahead_toe_norm",  # 최저점 무릎이 발끝보다 앞에 나간 거리, 다리 길이 단위
    "depth_ratio",  # 최저점 Pelvis 높이 / 선 자세 Pelvis 높이 (발목 기준)
)


def _p(seq: np.ndarray, name: str) -> np.ndarray:
    return seq[:, _IDX[name]]


def _elev_deg(vec: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(vec[..., 2], np.linalg.norm(vec[..., :2], axis=-1)))


def _incl_deg(vec: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(np.linalg.norm(vec[..., :2], axis=-1), vec[..., 2]))


def pelvis_height(aligned: np.ndarray) -> np.ndarray:
    """(T,) 발목 중점 대비 Pelvis 높이. Pelvis가 원점이므로 -발목 높이."""
    return -(_p(aligned, "LAnkle")[:, 2] + _p(aligned, "RAnkle")[:, 2]) / 2.0


def bottom_index(aligned: np.ndarray) -> int:
    return int(np.argmin(pelvis_height(aligned)))


def rep_features(aligned: np.ndarray, standing: np.ndarray) -> dict[str, float]:
    """반복 1회 (T, 18, 3)와 선 자세 마스크 (T,)로 ``FEATURE_NAMES`` 지표를 계산한다."""
    if not standing.any():
        raise ValueError("선 자세 프레임이 없습니다")
    b = bottom_index(aligned)
    height = pelvis_height(aligned)
    stand_h = float(np.median(height[standing]))
    deep = height <= height[b] + 0.5 * (stand_h - height[b])

    shoulder = (_p(aligned, "LShoulder") + _p(aligned, "RShoulder")) / 2.0
    trunk_incl = _incl_deg(shoulder)  # Pelvis가 원점
    per_side: dict[str, list[float]] = {k: [] for k in (
        "foot_angle_rise", "heel_rise_norm", "knee_min", "hip_min", "ankle_min",
        "thigh_elev_bottom", "tibia_incl_bottom", "trunk_minus_tibia_deep",
        "hip_knee_flex_ratio", "knee_ahead_toe_norm",
    )}
    legs = []
    for s in _SIDES:
        hip, knee, ankle = _p(aligned, f"{s}Hip"), _p(aligned, f"{s}Knee"), _p(aligned, f"{s}Ankle")
        heel, toe = _p(aligned, f"{s}Heel"), _p(aligned, f"{s}BigToe")
        leg = float(np.median(np.linalg.norm(hip - knee, axis=1) + np.linalg.norm(knee - ankle, axis=1)))
        legs.append(leg)

        foot = _elev_deg(heel - toe)
        per_side["foot_angle_rise"].append(foot.max() - np.median(foot[standing]))
        heel_above_toe = heel[:, 2] - toe[:, 2]
        per_side["heel_rise_norm"].append(
            (heel_above_toe.max() - np.median(heel_above_toe[standing])) / leg
        )
        knee_ang = joint_angle_deg(ankle, knee, hip)
        hip_ang = joint_angle_deg(knee, hip, shoulder)
        per_side["knee_min"].append(knee_ang.min())
        per_side["hip_min"].append(hip_ang.min())
        per_side["ankle_min"].append(joint_angle_deg(knee, ankle, toe).min())
        per_side["thigh_elev_bottom"].append(_elev_deg(hip[b] - knee[b]))
        tibia = _incl_deg(knee - ankle)
        per_side["tibia_incl_bottom"].append(tibia[b])
        per_side["trunk_minus_tibia_deep"].append(float(np.mean(trunk_incl[deep] - tibia[deep])))
        per_side["hip_knee_flex_ratio"].append((180.0 - hip_ang[b]) / max(180.0 - knee_ang[b], 1e-6))
        per_side["knee_ahead_toe_norm"].append((knee[b, 1] - toe[b, 1]) / leg)

    out = {k: float(np.mean(v)) for k, v in per_side.items()}
    leg = float(np.mean(legs))
    ankle_mid = (_p(aligned, "LAnkle") + _p(aligned, "RAnkle")) / 2.0
    out["trunk_incl_bottom"] = float(trunk_incl[b])
    out["trunk_minus_tibia"] = out["trunk_incl_bottom"] - out["tibia_incl_bottom"]
    out["hip_back_norm"] = float(ankle_mid[b, 1] / leg)
    out["depth_ratio"] = float(height[b] / stand_h)
    return {name: out[name] for name in FEATURE_NAMES}
