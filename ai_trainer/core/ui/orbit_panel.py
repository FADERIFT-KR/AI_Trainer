"""마우스 드래그로 시점을 돌려 보는 3D 스켈레톤 패널.

정면/사선/측면을 따로 띄우는 대신 한 화면에서 구(sphere) 위를 도는 카메라로 본다.
좌우 드래그는 수직축 둘레 회전(방위각), 위아래 드래그는 올려다보기/내려다보기(고도).
휠로 확대/축소하고, 더블클릭하면 정면으로 되돌아온다.

여기서 하는 일은 표시뿐이다 — 좌표를 바꾸지 않는다. 회전은 볼 각도만 바꾸고 판정에
들어가는 값에는 전혀 영향을 주지 않는다.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy

DEFAULT_AZIMUTH = 0.0    # 정면
DEFAULT_ELEVATION = 8.0  # 살짝 위에서 내려다본다 — 발끝이 겹쳐 보이지 않게
_MIN_ELEVATION, _MAX_ELEVATION = -85.0, 85.0
_MIN_ZOOM, _MAX_ZOOM = 0.4, 3.0
_DRAG_PER_DEGREE = 0.45  # 마우스 1픽셀당 회전 각도

_BG = (24, 28, 36)
_JOINT = (240, 240, 240)
_GRID = (44, 50, 62)
_ERROR = (0, 0, 255)  # 최종 판정에서 오류로 확인된 부위


class OrbitSkeletonPanel(QLabel):
    """(J,3) 좌표를 받아 현재 카메라 각도로 투영해 그린다."""

    def __init__(self, placeholder: str = "스켈레톤 준비 중…", parent=None) -> None:
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(360, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            "QLabel { background: #181c25; color: #9aa5b5; "
            "border: 1px solid #3b4352; border-radius: 8px; }"
        )
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip("드래그: 시점 회전 · 휠: 확대/축소 · 더블클릭: 정면으로")

        self.azimuth = DEFAULT_AZIMUTH
        self.elevation = DEFAULT_ELEVATION
        self.zoom = 1.0
        self._drag_from: tuple[int, int] | None = None

        self._points: np.ndarray | None = None
        self._bone_pairs: list[tuple[int, int]] = []
        self._bone_colors: list[tuple[int, int, int]] = []
        self._title = ""
        self._highlight: set[int] = set()   # 빨갛게 강조할 관절 인덱스
        self._footer = ""
        self._half_range = 1.0
        self._flip_vertical = True

    # ── 외부에서 넣어주는 것 ────────────────────────────────────────
    def configure(self, bone_pairs, bone_colors, half_range: float = 1.0,
                  flip_vertical: bool = True) -> None:
        """뼈대 연결/색과 표시 반경을 정한다.

        half_range는 화면에 담을 반지름(좌표 단위)이다. 프레임마다 자동으로 맞추면
        화면이 숨 쉬듯 확대·축소되어 실제 자세 변화와 구분이 안 되므로 고정해 둔다.
        """
        self._bone_pairs = list(bone_pairs)
        self._bone_colors = list(bone_colors)
        self._half_range = max(float(half_range), 1e-6)
        self._flip_vertical = flip_vertical

    def set_points(self, points_3d: np.ndarray | None, title: str = "",
                   highlight: set[int] | None = None, footer: str = "") -> None:
        self._points = None if points_3d is None else np.asarray(points_3d, dtype=float)
        self._title = title
        self._highlight = set(highlight or ())
        self._footer = footer
        self._repaint()

    def reset_view(self) -> None:
        self.azimuth, self.elevation, self.zoom = DEFAULT_AZIMUTH, DEFAULT_ELEVATION, 1.0
        self._repaint()

    # ── 마우스 ────────────────────────────────────────────────────
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_from = (event.x(), event.y())
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_from is None:
            return
        dx = event.x() - self._drag_from[0]
        dy = event.y() - self._drag_from[1]
        self._drag_from = (event.x(), event.y())
        self.azimuth = (self.azimuth + dx * _DRAG_PER_DEGREE) % 360.0
        self.elevation = max(_MIN_ELEVATION,
                             min(_MAX_ELEVATION, self.elevation + dy * _DRAG_PER_DEGREE))
        self._repaint()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_from = None
        self.setCursor(Qt.OpenHandCursor)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802
        self.reset_view()

    def wheelEvent(self, event) -> None:  # noqa: N802
        steps = event.angleDelta().y() / 120.0
        self.zoom = max(_MIN_ZOOM, min(_MAX_ZOOM, self.zoom * (1.12 ** steps)))
        self._repaint()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._repaint()

    # ── 그리기 ────────────────────────────────────────────────────
    def _project(self, points: np.ndarray, width: int, height: int) -> np.ndarray:
        """수직축 둘레로 azimuth, 그 다음 시선 축으로 elevation만큼 돌려 화면에 투영."""
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        horizontal = points[:, 0] * math.cos(az) + points[:, 2] * math.sin(az)
        depth = -points[:, 0] * math.sin(az) + points[:, 2] * math.cos(az)
        vertical = points[:, 1]
        # elevation: 위에서 내려다보면 깊이가 화면 세로로 섞여 들어온다
        screen_y = vertical * math.cos(el) - depth * math.sin(el)

        scale = min(width, height) * 0.46 * self.zoom / self._half_range
        x = horizontal * scale + width / 2.0
        y = (height / 2.0 - screen_y * scale if self._flip_vertical
             else height / 2.0 + screen_y * scale)
        return np.stack([x, y], axis=-1)

    def _repaint(self) -> None:
        width = max(self.width(), 80)
        height = max(self.height(), 80)
        canvas = np.full((height, width, 3), _BG, dtype=np.uint8)

        # 바닥 기준선 — 회전해도 위아래를 가늠할 수 있게
        cv2.line(canvas, (0, height // 2), (width, height // 2), _GRID, 1, cv2.LINE_AA)
        cv2.line(canvas, (width // 2, 0), (width // 2, height), _GRID, 1, cv2.LINE_AA)

        if self._points is not None and len(self._points):
            projected = self._project(self._points, width, height)
            valid = np.isfinite(projected).all(axis=1)
            for (i, j), color in zip(self._bone_pairs, self._bone_colors):
                if i < len(valid) and j < len(valid) and valid[i] and valid[j]:
                    stroke = _ERROR if (i in self._highlight or j in self._highlight) else color
                    cv2.line(canvas,
                             tuple(int(v) for v in np.round(projected[i])),
                             tuple(int(v) for v in np.round(projected[j])),
                             stroke, 2, cv2.LINE_AA)
            for i in range(len(valid)):
                if not valid[i]:
                    continue
                center = tuple(int(v) for v in np.round(projected[i]))
                if i in self._highlight:
                    cv2.circle(canvas, center, 9, _ERROR, -1, cv2.LINE_AA)
                    cv2.circle(canvas, center, 9, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    cv2.circle(canvas, center, 3, _JOINT, -1, cv2.LINE_AA)

        # 회전 중에도 지금 어느 각도인지 알 수 있게 항상 표시한다(OpenCV 폰트는 ASCII 전용).
        cv2.putText(canvas, f"az {self.azimuth:.0f}  el {self.elevation:.0f}  x{self.zoom:.2f}",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 175), 1, cv2.LINE_AA)
        if self._footer:
            cv2.putText(canvas, self._footer, (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 160, 175), 1, cv2.LINE_AA)

        rgb = np.ascontiguousarray(canvas[:, :, ::-1])
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(image))

    @property
    def view_caption(self) -> str:
        return f"방위 {self.azimuth:.0f}° · 고도 {self.elevation:.0f}° · 배율 {self.zoom:.2f}x"


__all__ = ["OrbitSkeletonPanel"]
