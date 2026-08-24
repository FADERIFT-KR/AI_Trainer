"""오류유형별 말풍선 설명: 해당 오류와 관련된 관절 옆에 짧은 문구를 표시한다.

관절/설명 매핑은 claude.md 9장(오류유형별 주요 feature)과 dtw_feature_weights.json의
class_overrides/feedback_focus_features를 따른다:
  - 발뒤꿈치오류 -> heel_height, ankle_angle
  - 엉덩이하방오류 -> pelvis_trajectory(최저점 깊이)
  - 고관절오류 -> torso_inclination, hip_flexion_angle
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


@dataclass(frozen=True)
class ErrorExplain:
    joints: tuple[str, ...]  # 말풍선을 붙일 기준 관절(들의 평균 위치)
    text: str


ERROR_EXPLANATIONS: dict[str, ErrorExplain] = {
    "발뒤꿈치오류": ErrorExplain(("LHeel", "RHeel"), "발뒤꿈치가 들려요"),
    "엉덩이하방오류": ErrorExplain(("Hip",), "덜 앉았어요, 더 내려가세요"),
    "고관절오류": ErrorExplain(("Hip", "Neck"), "상체가 많이 기울었어요"),
}


def anchor_point(common2d: np.ndarray, joints: tuple[str, ...]) -> tuple[int, int]:
    pts = np.array([common2d[_IDX[name]] for name in joints])
    x, y = pts.mean(axis=0)
    return int(x), int(y)


def draw_speech_bubble(
    img: np.ndarray,
    anchor: tuple[int, int],
    text: str,
    *,
    offset: tuple[int, int] = (40, -60),
    bg_color: tuple[int, int, int] = (40, 40, 235),
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """anchor(관절 픽셀 좌표) 옆에 말풍선(사각형+포인터+텍스트)을 그린다."""
    h, w = img.shape[:2]
    ax, ay = anchor
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.55, 1
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

    pad_x, pad_y = 10, 8
    box_w, box_h = tw + pad_x * 2, th + baseline + pad_y * 2

    bx = ax + offset[0]
    by = ay + offset[1]
    # 화면 밖으로 나가지 않도록 보정
    bx = max(4, min(bx, w - box_w - 4))
    by = max(4, min(by, h - box_h - 4))

    # 말풍선 몸통
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (bx, by), (bx + box_w, by + box_h), (255, 255, 255), 1, cv2.LINE_AA)

    # 관절까지 이어지는 포인터(삼각형)
    box_cx = bx + box_w // 2
    box_cy = by + box_h // 2
    pointer = np.array(
        [[ax, ay], [box_cx - 6, by + box_h if by + box_h < ay else by], [box_cx + 6, by + box_h if by + box_h < ay else by]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, pointer, bg_color, cv2.LINE_AA)

    # 관절 위치 강조 점
    cv2.circle(img, (ax, ay), 5, bg_color, -1, cv2.LINE_AA)
    cv2.circle(img, (ax, ay), 5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, text, (bx + pad_x, by + pad_y + th), font, scale, text_color, thickness, cv2.LINE_AA)


def annotate_error(video_bgr: np.ndarray, common2d: np.ndarray, error_class: str) -> None:
    """error_class(정상이 아닌 오류유형)에 해당하는 말풍선을 video_bgr에 그려 넣는다(in-place)."""
    explain = ERROR_EXPLANATIONS.get(error_class)
    if explain is None:
        return
    anchor = anchor_point(common2d, explain.joints)
    draw_speech_bubble(video_bgr, anchor, explain.text)
