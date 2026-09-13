"""실시간 스쿼트 기본 자세 조건 검사.

카메라 정면 영상에서 안정적으로 관측할 수 있는 발/무릎/골반 관계만 검사한다.
거리 판정은 이 모듈에서 수행하지 않고 framing_check에서 준비 자세에 한정한다.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# MediaPipe Pose landmark indices.
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
L_HIP, R_HIP, L_KNEE, R_KNEE = 23, 24, 25, 26
L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_TOE, R_TOE = 27, 28, 29, 30, 31, 32


@dataclass(frozen=True)
class FormAssessment:
    warnings: tuple[str, ...] = ()
    foot_angles: tuple[float, float] | None = None  # left(+CCW), right(-CW); camera direction=0°


def _point(p: np.ndarray, i: int, width: int, height: int) -> np.ndarray:
    return np.asarray([p[i, 0] * width, p[i, 1] * height], dtype=np.float64)


def _visible(p: np.ndarray, indices: tuple[int, ...], threshold: float = 0.4) -> bool:
    return all(np.isfinite(p[i, :3]).all() and float(p[i, 3]) >= threshold for i in indices)


def _outward_toe_angle(knee: np.ndarray, ankle: np.ndarray, toe: np.ndarray, side: int) -> float:
    """정강이뼈(무릎→발목)를 0도로 둔 발끝 외회전 각도(도).

    좌우 방향이 반대이므로 ``side``(+1 왼쪽, -1 오른쪽)를 곱해
    바깥쪽 회전만 양수로 정규화한다.  영상에서 발이 거의 옆으로 보이는
    경우에도 180도 보정으로 가장 가까운 상대각을 사용한다.
    """
    shin = ankle - knee
    foot = toe - ankle
    shin_norm, foot_norm = np.linalg.norm(shin), np.linalg.norm(foot)
    if shin_norm < 1e-6 or foot_norm < 1e-6:
        return 90.0
    signed = math.degrees(
        math.atan2(float(shin[0] * foot[1] - shin[1] * foot[0]), float(np.dot(shin, foot)))
    )
    while signed > 90.0:
        signed -= 180.0
    while signed < -90.0:
        signed += 180.0
    outward = side * signed
    return outward if outward >= 0.0 else 180.0 + outward


def _camera_relative_toe_angle(ankle: np.ndarray, toe: np.ndarray, side: int) -> float:
    """발목→엄지발가락 벡터를 카메라 방향 기준으로 독립 계산한다.

    MediaPipe world ``-z``(카메라 방향)를 0도로 둔다. 객체가 우측으로
    45도 회전하면 우측 발은 +45도, 좌측 발은 -45도가 된다.
    객체(인물) 기준 좌측은 음수, 우측은 양수로 표시하며 다른 발의 좌표는
    전혀 사용하지 않는다. ``side``는 객체 기준 좌측 -1, 우측 +1이다.
    """
    vec = np.asarray(toe, dtype=np.float64) - np.asarray(ankle, dtype=np.float64)
    lateral, depth_toward_camera = float(vec[0]), -float(vec[2])
    magnitude = abs(math.degrees(math.atan2(lateral, depth_toward_camera)))
    # 카메라 영상의 좌우(미러링 포함)가 아니라 해부학적 L/R 라벨로 부호를 부여한다.
    return float(side * magnitude)


def _xz_vector_angle(first: np.ndarray, second: np.ndarray) -> float:
    """두 3D 벡터의 x-z 지면 투영 방향 차이(0~180도)."""
    a = np.asarray(first, dtype=np.float64)[[0, 2]]
    b = np.asarray(second, dtype=np.float64)[[0, 2]]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 180.0
    cosine = float(np.dot(a, b) / (na * nb))
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def assess_squat_form(
    image_landmarks: np.ndarray,
    width: int,
    height: int,
    phase: str,
    world_landmarks: np.ndarray | None = None,
) -> FormAssessment:
    """준비/하강/최저점/상승 phase별 기본 자세 경고를 반환한다."""
    p = np.asarray(image_landmarks, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 33 or p.shape[1] < 4:
        return FormAssessment(("관절 데이터를 충분히 인식하지 못했어요",), None)

    warnings: list[str] = []
    feet = (L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_TOE, R_TOE)
    if not _visible(p, feet):
        return FormAssessment(("발이 잘 보이지 않아요 — 발 전체가 화면에 보이게 해주세요",), None)

    ls, rs = _point(p, L_SHOULDER, width, height), _point(p, R_SHOULDER, width, height)
    lh, rh = _point(p, L_HIP, width, height), _point(p, R_HIP, width, height)
    la, ra = _point(p, L_ANKLE, width, height), _point(p, R_ANKLE, width, height)
    lheel, rheel = _point(p, L_HEEL, width, height), _point(p, R_HEEL, width, height)
    ltoe, rtoe = _point(p, L_TOE, width, height), _point(p, R_TOE, width, height)
    lk, rk = _point(p, L_KNEE, width, height), _point(p, R_KNEE, width, height)

    shoulder_w = max(float(np.linalg.norm(ls - rs)), 1.0)
    hip_w = max(float(np.linalg.norm(lh - rh)), 1.0)
    foot_w = float(np.linalg.norm(la - ra))
    if world_landmarks is not None and np.asarray(world_landmarks).shape[0] >= 33:
        world = np.asarray(world_landmarks, dtype=np.float64)
        left_angle = _camera_relative_toe_angle(world[L_ANKLE, :3], world[L_TOE, :3], -1)
        right_angle = _camera_relative_toe_angle(world[R_ANKLE, :3], world[R_TOE, :3], 1)
    else:
        # 테스트/구형 호출 호환용 2D fallback
        left_angle = -_outward_toe_angle(lk, la, ltoe, 1)
        right_angle = _outward_toe_angle(rk, ra, rtoe, -1)

    if phase == "prep":
        lower, upper = 0.75 * min(shoulder_w, hip_w), 1.55 * max(shoulder_w, hip_w)
        if not lower <= foot_w <= upper:
            warnings.append("발 간격을 어깨 너비~골반 너비에 맞춰주세요")

        if not -45.0 <= left_angle <= -10.0:
            warnings.append("왼발 각도를 정면 기준 -45~-10도로 맞춰주세요")
        if not 10.0 <= right_angle <= 45.0:
            warnings.append("오른발 각도를 정면 기준 +10~+45도로 맞춰주세요")

    # Heel/toe 높이 차가 크면 발바닥 전체가 지면에 닿지 않은 것으로 간주한다.
    shoulder_y = float((ls[1] + rs[1]) / 2.0)
    heel_y = float((lheel[1] + rheel[1]) / 2.0)
    body_h = max(abs(shoulder_y - heel_y), height * 0.35)
    contact_tol = 0.10 * body_h
    if abs(float(lheel[1] - ltoe[1])) > contact_tol or abs(float(rheel[1] - rtoe[1])) > contact_tol:
        warnings.append("발바닥 전체가 지면에 닿도록 해주세요")

    if phase == "bottom":
        # 이미지 y가 클수록 아래쪽이므로 골반이 무릎 높이 이하(수평 또는 더 낮음)인지 확인한다.
        hip_y = float((lh[1] + rh[1]) / 2.0)
        knee_y = float((lk[1] + rk[1]) / 2.0)
        if hip_y < knee_y - 0.03 * body_h:
            warnings.append("엉덩이를 무릎 높이까지 또는 그보다 낮게 내려가세요")

        # 정면 영상에서 확인 가능한 고관절 접기/상체 기울기 대체 지표.
        shoulder_mid, hip_mid = (ls + rs) / 2.0, (lh + rh) / 2.0
        torso = shoulder_mid - hip_mid
        torso_angle = abs(math.degrees(math.atan2(float(torso[0]), float(-torso[1] + 1e-9))))
        if torso_angle < 8.0:
            warnings.append("엉덩이를 뒤로 빼며 고관절을 접어 내려가세요")

    return FormAssessment(tuple(dict.fromkeys(warnings)), (left_angle, right_angle))


__all__ = ["FormAssessment", "assess_squat_form"]
