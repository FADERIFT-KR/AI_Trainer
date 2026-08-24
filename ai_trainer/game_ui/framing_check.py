"""사용자가 카메라 화각 안에 '분석하기 좋은 위치/자세'로 서 있는지 실시간 점검.

DTW 비교는 AI Hub 정면 카메라(camera1) 데이터로 학습·구축되었으므로, 실사용자도
같은 조건(전신이 화면에 들어오고, 카메라와 적당한 거리를 두고, 정면을 보고 서 있음)일
때만 비교 결과를 신뢰할 수 있다. 이 체크를 통과하지 못하면 OnlineSquatSession에
프레임을 넣지 않는다(잘못된 phase/DTW 상태가 쌓이는 것을 방지).

정면 판별 기준은 claude.md 7장에서 camera1을 정면 후보로 검증할 때 쓴 것과 동일한
휴리스틱(어깨/골반의 좌우 분리 정도)을 재사용한다 — 학습에 쓰인 AI Hub 정면 카메라와
같은 뷰를 사용자에게 요구해야 lifting 모델의 도메인과 어긋나지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# image_landmarks(33,4) 인덱스
NOSE, L_SHOULDER, R_SHOULDER = 0, 11, 12
L_HIP, R_HIP, L_KNEE, R_KNEE = 23, 24, 25, 26
L_ANKLE, R_ANKLE, L_FOOT, R_FOOT = 27, 28, 31, 32

REQUIRED_LANDMARKS = (NOSE, L_SHOULDER, R_SHOULDER, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_FOOT, R_FOOT)

# 프레임 대비 이상적인 전신 bbox 비율 (정면에서 2~3m 거리 기준 목표치)
MIN_BODY_HEIGHT_RATIO = 0.55
MAX_BODY_HEIGHT_RATIO = 0.90
EDGE_MARGIN_RATIO = 0.03
CENTER_TOLERANCE_RATIO = 0.18
MIN_VISIBILITY = 0.5
MIN_FRONTAL_HIP_RATIO = 0.10  # |LHip_x-RHip_x| / body_bbox_width, 이보다 작으면 옆모습으로 판단


@dataclass(frozen=True)
class FramingResult:
    ok: bool
    message: str
    guide_box: tuple[int, int, int, int]  # 화면에 그릴 목표 영역 (x0,y0,x1,y1)
    body_box: tuple[int, int, int, int] | None  # 실제 감지된 전신 bbox (있으면)


def guide_box(width: int, height: int) -> tuple[int, int, int, int]:
    x0 = int(width * (0.5 - (MAX_BODY_HEIGHT_RATIO * 0.35)))
    x1 = int(width * (0.5 + (MAX_BODY_HEIGHT_RATIO * 0.35)))
    y0 = int(height * (1 - MAX_BODY_HEIGHT_RATIO) / 2)
    y1 = int(height * (1 + MAX_BODY_HEIGHT_RATIO) / 2)
    return x0, y0, x1, y1


def check_framing(image_landmarks: np.ndarray, width: int, height: int) -> FramingResult:
    box = guide_box(width, height)
    xs = image_landmarks[:, 0] * width
    ys = image_landmarks[:, 1] * height
    vis = image_landmarks[:, 3]

    missing = [i for i in REQUIRED_LANDMARKS if vis[i] < MIN_VISIBILITY]
    if missing:
        if any(i in (L_ANKLE, R_ANKLE, L_FOOT, R_FOOT) for i in missing):
            msg = "발이 화면에 안 보여요 — 카메라에서 조금 더 멀어져 전신이 다 보이게 서주세요"
        elif NOSE in missing:
            msg = "얼굴이 안 보여요 — 카메라를 정면으로 봐주세요"
        else:
            msg = "몸 일부가 화면 밖에 있어요 — 전신이 다 보이도록 위치를 조정해주세요"
        return FramingResult(False, msg, box, None)

    used = REQUIRED_LANDMARKS
    x0b, x1b = float(xs[list(used)].min()), float(xs[list(used)].max())
    y0b, y1b = float(ys[list(used)].min()), float(ys[list(used)].max())
    body_box = (int(x0b), int(y0b), int(x1b), int(y1b))
    body_h = y1b - y0b
    body_w = max(x1b - x0b, 1.0)

    if x0b <= width * EDGE_MARGIN_RATIO or x1b >= width * (1 - EDGE_MARGIN_RATIO):
        return FramingResult(False, "몸이 화면 가장자리에 걸려있어요 — 카메라에서 조금 물러나주세요", box, body_box)
    if y0b <= height * EDGE_MARGIN_RATIO or y1b >= height * (1 - EDGE_MARGIN_RATIO):
        return FramingResult(False, "머리나 발이 화면에 잘려요 — 카메라에서 조금 물러나주세요", box, body_box)

    height_ratio = body_h / height
    if height_ratio < MIN_BODY_HEIGHT_RATIO:
        return FramingResult(False, "카메라에 조금 더 가까이 서주세요", box, body_box)
    if height_ratio > MAX_BODY_HEIGHT_RATIO:
        return FramingResult(False, "카메라에서 조금 더 물러나주세요", box, body_box)

    center_x = (x0b + x1b) / 2
    if center_x < width * (0.5 - CENTER_TOLERANCE_RATIO):
        return FramingResult(False, "오른쪽으로 조금 이동해주세요", box, body_box)
    if center_x > width * (0.5 + CENTER_TOLERANCE_RATIO):
        return FramingResult(False, "왼쪽으로 조금 이동해주세요", box, body_box)

    hip_sep = abs(xs[L_HIP] - xs[R_HIP])
    if hip_sep / body_w < MIN_FRONTAL_HIP_RATIO:
        return FramingResult(False, "카메라를 정면으로 봐주세요 (옆모습으로는 분석이 어려워요)", box, body_box)

    return FramingResult(True, "준비 완료 — 스쿼트를 시작하세요", box, body_box)
