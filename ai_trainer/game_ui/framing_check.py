"""사용자가 요청된 정면/좌측면/우측면 화각에 서 있는지 실시간 점검."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ai_trainer.camera_views import VIEW_FRONT, VIEW_LABEL_KO, VIEW_LEFT, VIEW_RIGHT, VIEWS
from ai_trainer.view_conditions import load_view_condition_config

# image_landmarks(33,4) 인덱스
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
L_HIP, R_HIP, L_KNEE, R_KNEE = 23, 24, 25, 26
L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_FOOT, R_FOOT = 27, 28, 29, 30, 31, 32

# BigToe(foot_index)는 MediaPipe에서 가장 불안정하게 잡히는 랜드마크라(작고, 신발/각도에
# 취약) 필수 조건에서 뺐다 — 발이 화면에 들어왔는지는 ankle/heel로도 충분히 판단 가능.
REQUIRED_LANDMARKS = (NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_HEEL, R_HEEL)
REQUIRED_BY_VIEW = {
    VIEW_FRONT: REQUIRED_LANDMARKS,
    # 반대편 무릎/발목은 측면에서 자기 가림이 크므로 필수에서 제외한다. 어깨와
    # 골반 양쪽은 시점 확인을 위해 유지한다.
    VIEW_LEFT: (NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, L_ANKLE, L_HEEL),
    VIEW_RIGHT: (NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, R_KNEE, R_ANKLE, R_HEEL),
}
REQUIRED_LANDMARK_NAMES = {
    NOSE: "코", L_SHOULDER: "왼어깨", R_SHOULDER: "오른어깨",
    L_HIP: "왼골반", R_HIP: "오른골반", L_KNEE: "왼무릎", R_KNEE: "오른무릎",
    L_ANKLE: "왼발목", R_ANKLE: "오른발목", L_HEEL: "왼뒤꿈치", R_HEEL: "오른뒤꿈치",
}

# 프레임 대비 이상적인 전신 bbox 비율 (정면에서 2~3m 거리 기준 목표치)
MIN_BODY_HEIGHT_RATIO = 0.50
MAX_BODY_HEIGHT_RATIO = 0.92
EDGE_MARGIN_RATIO = 0.02
CENTER_TOLERANCE_RATIO = 0.20
MIN_VISIBILITY = 0.4
MIN_FRONTAL_HIP_RATIO = 0.07  # |LHip_x-RHip_x| / body_bbox_width, 이보다 작으면 옆모습으로 판단
MAX_SIDE_HIP_RATIO = 0.20
UPRIGHT_KNEE_ANGLE_DEG = 165.0
CALIBRATION_TRUNK_TILT_DEG = 15.0


def _dataset_view_threshold() -> float | None:
    """Return the train-normal midpoint between frontal and side hip separation."""
    try:
        config = load_view_condition_config()
        key = "준비|hip_screen_separation_ratio|median"
        front = float(config["normal_ranges"][VIEW_FRONT][key]["median"])
        side = max(
            float(config["normal_ranges"][VIEW_LEFT][key]["median"]),
            float(config["normal_ranges"][VIEW_RIGHT][key]["median"]),
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(front) or not np.isfinite(side) or front <= side:
        return None
    return (front + side) / 2.0


_DATASET_VIEW_THRESHOLD = _dataset_view_threshold()


@dataclass(frozen=True)
class FramingResult:
    ok: bool
    message: str
    guide_box: tuple[int, int, int, int]  # 화면에 그릴 목표 영역 (x0,y0,x1,y1)
    body_box: tuple[int, int, int, int] | None  # 실제 감지된 전신 bbox (있으면)
    low_confidence_joints: tuple[tuple[str, float], ...] = ()  # 진단용: (관절명, visibility) 임계값 미달 목록


@dataclass(frozen=True)
class UprightCalibrationResult:
    """Whether this frame is safe to use as the fixed body-axis baseline."""

    ok: bool
    message: str
    knee_angle_deg: float | None = None
    trunk_tilt_deg: float | None = None


def guide_box(width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(width * (0.5 - (MAX_BODY_HEIGHT_RATIO * 0.35)))
    x1 = int(width * (0.5 + (MAX_BODY_HEIGHT_RATIO * 0.35)))
    y0 = int(height * (1 - MAX_BODY_HEIGHT_RATIO) / 2)
    y1 = int(height * (1 + MAX_BODY_HEIGHT_RATIO) / 2)
    return x0, y0, x1, y1


def upright_calibration_pose(image_landmarks: np.ndarray, view: str) -> UprightCalibrationResult:
    """Accept only an upright, extended pose for fixed 3D axis calibration.

    The fixed body coordinate frame must be derived before a squat begins.  A
    knee can be bent while moving even when the camera framing is valid, so the
    framing gate alone is not enough.  The close leg is used in side views;
    the two legs are both required in the frontal view.
    """
    if view not in VIEWS:
        raise ValueError(f"지원하지 않는 시점입니다: {view}")
    points = np.asarray(image_landmarks, dtype=float)
    if points.shape != (33, 4) or not np.isfinite(points[:, :2]).all():
        return UprightCalibrationResult(False, "선 자세 보정 대기 중 — 관절을 다시 인식해주세요")

    if view == VIEW_FRONT:
        legs = ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
        shoulder = (points[L_SHOULDER, :2] + points[R_SHOULDER, :2]) / 2.0
        hip = (points[L_HIP, :2] + points[R_HIP, :2]) / 2.0
    elif view == VIEW_LEFT:
        legs = ((L_HIP, L_KNEE, L_ANKLE),)
        shoulder, hip = points[L_SHOULDER, :2], points[L_HIP, :2]
    else:
        legs = ((R_HIP, R_KNEE, R_ANKLE),)
        shoulder, hip = points[R_SHOULDER, :2], points[R_HIP, :2]

    angles = []
    for hip_i, knee_i, ankle_i in legs:
        if min(points[hip_i, 3], points[knee_i, 3], points[ankle_i, 3]) < MIN_VISIBILITY:
            return UprightCalibrationResult(False, "선 자세 보정 대기 중 — 다리 관절을 화면에 보여주세요")
        upper, lower = points[hip_i, :2] - points[knee_i, :2], points[ankle_i, :2] - points[knee_i, :2]
        denominator = float(np.linalg.norm(upper) * np.linalg.norm(lower))
        if denominator <= 1e-6:
            return UprightCalibrationResult(False, "선 자세 보정 대기 중 — 다리 관절을 다시 인식해주세요")
        angles.append(float(np.degrees(np.arccos(np.clip(np.dot(upper, lower) / denominator, -1.0, 1.0)))))
    knee_angle = min(angles)
    if knee_angle < UPRIGHT_KNEE_ANGLE_DEG:
        return UprightCalibrationResult(
            False, "선 자세 보정 대기 중 — 무릎을 펴고 잠시 서주세요", knee_angle_deg=knee_angle,
        )

    if min(points[L_SHOULDER, 3], points[R_SHOULDER, 3], points[L_HIP, 3], points[R_HIP, 3]) < MIN_VISIBILITY:
        return UprightCalibrationResult(False, "선 자세 보정 대기 중 — 어깨와 골반을 화면에 보여주세요",
                                       knee_angle_deg=knee_angle)
    trunk = shoulder - hip
    trunk_tilt = float(np.degrees(np.arctan2(abs(trunk[0]), abs(trunk[1]))))
    if trunk_tilt > CALIBRATION_TRUNK_TILT_DEG:
        return UprightCalibrationResult(
            False, "선 자세 보정 대기 중 — 상체를 세우고 잠시 서주세요",
            knee_angle_deg=knee_angle, trunk_tilt_deg=trunk_tilt,
        )
    return UprightCalibrationResult(
        True, "선 자세 보정 중 — 잠시 그대로 서주세요",
        knee_angle_deg=knee_angle, trunk_tilt_deg=trunk_tilt,
    )


def check_framing(
    image_landmarks: np.ndarray,
    width: int,
    height: int,
    relax_distance: bool = False,
    view: str = VIEW_FRONT,
) -> FramingResult:
    """활동 중에는 준비 자세에서만 유효한 거리/시점 추정치를 다시 검사하지 않는다.

    스쿼트에서 다리가 화면상 짧아지면 hip separation / projected leg length가
    측면 자세에서도 정면 수준으로 상승한다. 저장된 실사용 우측면 녹화에서는
    최저점 근처 75/182프레임이 방향 오류로 버려져 3D 추적과 phase가 끊겼다.
    시점은 카운트다운 전과 활동 중 똑바로 선 구간에서 확인한다. 무릎이
    굽혀진 동안에는 가시성/화면 경계/중앙만 확인한다.
    """
    if view not in VIEWS:
        raise ValueError(f"지원하지 않는 촬영 시점입니다: {view}")
    box = guide_box(width, height)
    xs = image_landmarks[:, 0] * width
    ys = image_landmarks[:, 1] * height
    vis = image_landmarks[:, 3]

    # 진단용: 임계값 미달인 필수 관절과 그 visibility 값 — 왜 안 되는지 화면에 바로 보여주기 위함
    required = REQUIRED_BY_VIEW[view]
    low_conf = tuple(
        (REQUIRED_LANDMARK_NAMES[i], float(vis[i])) for i in required if vis[i] < MIN_VISIBILITY
    )

    missing = [i for i in required if vis[i] < MIN_VISIBILITY]
    if missing:
        if any(i in (L_ANKLE, R_ANKLE, L_HEEL, R_HEEL) for i in missing):
            msg = "발이 화면에 안 보여요 — 카메라에서 조금 더 멀어져 전신이 다 보이게 서주세요"
        elif NOSE in missing:
            msg = f"얼굴이 안 보여요 — {VIEW_LABEL_KO[view]} 방향을 유지해주세요"
        else:
            msg = "몸 일부가 화면 밖에 있어요 — 전신이 다 보이도록 위치를 조정해주세요"
        return FramingResult(False, msg, box, None, low_conf)

    used = required
    x0b, x1b = float(xs[list(used)].min()), float(xs[list(used)].max())
    y0b, y1b = float(ys[list(used)].min()), float(ys[list(used)].max())
    body_box = (int(x0b), int(y0b), int(x1b), int(y1b))
    body_h = y1b - y0b
    body_w = max(x1b - x0b, 1.0)

    if x0b <= width * EDGE_MARGIN_RATIO or x1b >= width * (1 - EDGE_MARGIN_RATIO):
        return FramingResult(False, "몸이 화면 가장자리에 걸려있어요 — 카메라에서 조금 물러나주세요", box, body_box, low_conf)
    if y0b <= height * EDGE_MARGIN_RATIO or y1b >= height * (1 - EDGE_MARGIN_RATIO):
        return FramingResult(False, "머리나 발이 화면에 잘려요 — 카메라에서 조금 물러나주세요", box, body_box, low_conf)

    if not relax_distance:
        # 웹캠 전용 브랜치: 가로(webcam) 프레임 기준으로 고정. 세로(휴대폰) 촬영
        # 대응(min(width,height) 정규화)은 feature/game-ui-phone에서만 유지한다.
        height_ratio = body_h / height
        if height_ratio < MIN_BODY_HEIGHT_RATIO:
            return FramingResult(False, "카메라에 조금 더 가까이 서주세요", box, body_box, low_conf)
        if height_ratio > MAX_BODY_HEIGHT_RATIO:
            return FramingResult(False, "카메라에서 조금 더 물러나주세요", box, body_box, low_conf)

    center_x = (x0b + x1b) / 2
    if center_x < width * (0.5 - CENTER_TOLERANCE_RATIO):
        return FramingResult(False, "오른쪽으로 조금 이동해주세요", box, body_box, low_conf)
    if center_x > width * (0.5 + CENTER_TOLERANCE_RATIO):
        return FramingResult(False, "왼쪽으로 조금 이동해주세요", box, body_box, low_conf)

    hip_sep = abs(xs[L_HIP] - xs[R_HIP])
    if relax_distance:
        # 힙 간격/화면상 다리 길이 비율은 무릎이 접힐 때 시점이
        # 변하지 않아도 크게 바뀐다. 선 자세에서만 시점을 재검사한다.
        side = (L_HIP, L_KNEE, L_ANKLE) if view == VIEW_LEFT else (R_HIP, R_KNEE, R_ANKLE)
        if view == VIEW_FRONT:
            pairs = ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
        else:
            pairs = (side,)
        knee_angles = []
        for hip, knee, ankle in pairs:
            upper = np.array([xs[hip] - xs[knee], ys[hip] - ys[knee]])
            lower = np.array([xs[ankle] - xs[knee], ys[ankle] - ys[knee]])
            denom = float(np.linalg.norm(upper) * np.linalg.norm(lower))
            if denom > 1e-6:
                knee_angles.append(float(np.degrees(np.arccos(np.clip(
                    np.dot(upper, lower) / denom, -1.0, 1.0
                )))))
        if knee_angles and min(knee_angles) < UPRIGHT_KNEE_ANGLE_DEG:
            return FramingResult(True, f"{VIEW_LABEL_KO[view]} 측정 중", box, body_box, low_conf)
    if _DATASET_VIEW_THRESHOLD is not None:
        leg_scale = np.mean(
            [
                np.hypot(xs[L_HIP] - xs[L_KNEE], ys[L_HIP] - ys[L_KNEE])
                + np.hypot(xs[L_KNEE] - xs[L_ANKLE], ys[L_KNEE] - ys[L_ANKLE]),
                np.hypot(xs[R_HIP] - xs[R_KNEE], ys[R_HIP] - ys[R_KNEE])
                + np.hypot(xs[R_KNEE] - xs[R_ANKLE], ys[R_KNEE] - ys[R_ANKLE]),
            ]
        )
        orientation_ratio = hip_sep / max(float(leg_scale), 1.0)
        # AI Hub 수동 keypoint와 MediaPipe 사이 도메인 차이를 고려해 midpoint 주변에
        # 25% 중첩 허용구간을 둔다. 명백히 다른 시점만 거부한다.
        if view == VIEW_FRONT and orientation_ratio < _DATASET_VIEW_THRESHOLD * 0.75:
            return FramingResult(False, "카메라를 정면으로 봐주세요", box, body_box, low_conf)
        if view != VIEW_FRONT and orientation_ratio > _DATASET_VIEW_THRESHOLD * 1.25:
            return FramingResult(False, f"{VIEW_LABEL_KO[view]}이 카메라를 향하도록 몸을 돌려주세요", box, body_box, low_conf)
    elif view == VIEW_FRONT and hip_sep / body_w < MIN_FRONTAL_HIP_RATIO:
        return FramingResult(False, "카메라를 정면으로 봐주세요", box, body_box, low_conf)
    elif view != VIEW_FRONT and hip_sep / body_w > MAX_SIDE_HIP_RATIO:
        return FramingResult(False, f"{VIEW_LABEL_KO[view]}이 카메라를 향하도록 몸을 돌려주세요", box, body_box, low_conf)

    return FramingResult(True, f"{VIEW_LABEL_KO[view]} 준비 완료 — 스쿼트를 시작하세요", box, body_box, low_conf)
