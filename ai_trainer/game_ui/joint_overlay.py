"""joint_feedback.JointScore 결과를 웹캠 화면 위 관절 점 색깔로 표시.

우측 막대바(screens.py)를 보지 않아도 어느 관절이 문제인지 스켈레톤 위에서 바로
보이도록, 각 관절의 색을 상태(good/warning/bad)에 따라 초록/노랑/빨강으로 그린다.
error_explain.annotate_error(말풍선)와 같은 패턴 — video_bgr을 in-place로 수정한다.
"""
from __future__ import annotations

import cv2
import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.joint_feedback import STATUS_BAD, STATUS_GOOD, STATUS_WARNING, JointScore

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}

# BGR (OpenCV) — 프로젝트 전반에서 쓰는 초록/빨강 톤과 맞춤(pipeline_worker.py 참고).
_STATUS_COLOR_BGR: dict[str, tuple[int, int, int]] = {
    STATUS_GOOD: (90, 220, 90),
    STATUS_WARNING: (0, 210, 255),
    STATUS_BAD: (60, 60, 240),
}


def draw_joint_feedback(video_bgr: np.ndarray, common2d: np.ndarray, joint_scores: list[JointScore]) -> None:
    """common2d(공통 스켈레톤 18관절 픽셀 좌표)를 기준으로, joint_scores에 담긴 관절
    위치에 상태별 색깔의 원을 그린다."""
    for js in joint_scores:
        idx = _IDX.get(js.common_joint)
        if idx is None:
            continue
        x, y = common2d[idx]
        color = _STATUS_COLOR_BGR.get(js.status, (200, 200, 200))
        center = (int(x), int(y))
        cv2.circle(video_bgr, center, 9, color, -1, cv2.LINE_AA)
        cv2.circle(video_bgr, center, 9, (255, 255, 255), 1, cv2.LINE_AA)
