"""2D/3D 스켈레톤 좌표를 OpenCV 프레임으로 그리는 렌더링 유틸리티."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .skeleton import BONE_COLORS_BGR, BONE_INDEX_PAIRS

TransformFn = Callable[[np.ndarray], np.ndarray]


def fit_transform(
    coords_seq: np.ndarray,
    panel_w: int,
    panel_h: int,
    margin: int = 30,
    flip_y: bool = True,
) -> TransformFn:
    """시퀀스 전체 (T, J, 2) 좌표 범위를 기준으로 패널 크기에 맞추는 변환 함수를 만든다.

    프레임마다 범위를 다시 계산하면 화면이 계속 확대/축소되어 흔들리므로,
    시퀀스 전체 min/max를 기준으로 스케일을 한 번만 고정한다.
    가로세로 비율(aspect ratio)은 유지한다(min(scale_x, scale_y) 사용).
    """
    xs = coords_seq[..., 0]
    ys = coords_seq[..., 1]
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    x_range = max(x_max - x_min, 1e-3)
    y_range = max(y_max - y_min, 1e-3)

    avail_w = panel_w - 2 * margin
    avail_h = panel_h - 2 * margin
    scale = min(avail_w / x_range, avail_h / y_range)

    # 스케일 적용 후 실제 차지하는 크기만큼 패널 중앙에 배치
    used_w = x_range * scale
    used_h = y_range * scale
    off_x = margin + (avail_w - used_w) / 2.0
    off_y = margin + (avail_h - used_h) / 2.0

    def transform(points: np.ndarray) -> np.ndarray:
        px = (points[..., 0] - x_min) * scale + off_x
        py = (points[..., 1] - y_min) * scale + off_y
        if flip_y:
            py = panel_h - py
        return np.stack([px, py], axis=-1)

    return transform


def draw_skeleton_panel(
    canvas: np.ndarray,
    origin_xy: tuple[int, int],
    panel_w: int,
    panel_h: int,
    points_px: np.ndarray,
    title: str,
    footer: str | None = None,
    bone_pairs: list[tuple[int, int]] | None = None,
    bone_colors: list[tuple[int, int, int]] | None = None,
) -> None:
    """canvas의 (origin_xy) 위치에 panel_w x panel_h 크기로 스켈레톤 한 프레임을 그린다.

    points_px: 이미 fit_transform으로 픽셀 좌표로 변환된 (J, 2) 배열.
    bone_pairs/bone_colors를 지정하지 않으면 AI Hub 26관절 기본 뼈대를 사용한다.
    """
    if bone_pairs is None:
        bone_pairs = BONE_INDEX_PAIRS
    if bone_colors is None:
        bone_colors = BONE_COLORS_BGR

    x0, y0 = origin_xy
    panel = np.full((panel_h, panel_w, 3), 24, dtype=np.uint8)

    valid = np.isfinite(points_px).all(axis=-1)

    for (i, j), color in zip(bone_pairs, bone_colors):
        if not (valid[i] and valid[j]):
            continue
        p1 = (int(round(points_px[i, 0])), int(round(points_px[i, 1])))
        p2 = (int(round(points_px[j, 0])), int(round(points_px[j, 1])))
        cv2.line(panel, p1, p2, color, 2, cv2.LINE_AA)

    for i in range(points_px.shape[0]):
        if not valid[i]:
            continue
        p = (int(round(points_px[i, 0])), int(round(points_px[i, 1])))
        cv2.circle(panel, p, 3, (255, 255, 255), -1, cv2.LINE_AA)

    cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (70, 70, 70), 1)
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
    if footer:
        cv2.putText(panel, footer, (8, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    canvas[y0 : y0 + panel_h, x0 : x0 + panel_w] = panel
