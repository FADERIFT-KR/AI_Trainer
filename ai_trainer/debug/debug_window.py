"""10패널 디버그 창 — 공정별 스켈레톤을 나란히 그리고 프레임 간 흔들림을 수치로 보여준다."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from ai_trainer.core.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS
from ai_trainer.core.live_pose.render import POSE_CONNECTIONS
from ai_trainer.debug.stages import STAGES, StageDef, StageSnapshot

DRAW_W, DRAW_H = 320, 230
COLUMNS = 5

_BG = (22, 26, 34)
_JOINT = (240, 240, 240)
_MP_BONE = (200, 200, 90)

# 3D 패널의 고정 표시 범위(hip 중심 반경). 프레임마다 재계산하면 화면이 숨쉬듯 흔들려
# 진짜 지터와 구분이 안 되므로 단위별로 고정한다.
_HALF_RANGE = {"mp3d": 1.0, "m3d": 1.0, "norm3d": 1.7}
_FLIP_Y = {"mp3d": False, "m3d": False, "norm3d": True}

# 지터 경고 기준(프레임 간 관절 최대 이동량의 30프레임 이동평균)
_JITTER_WARN = {"px": 6.0, "m": 0.02, "L": 0.05}
_JITTER_BAD = {"px": 15.0, "m": 0.05, "L": 0.12}
_COLOR_OK, _COLOR_WARN, _COLOR_BAD, _COLOR_MUTE = "#7fdc7f", "#ffb14e", "#ff5c5c", "#8a93a3"


class _StagePanel(QWidget):
    def __init__(self, index: int, stage: StageDef) -> None:
        super().__init__()
        self.stage = stage
        self._prev: np.ndarray | None = None
        self._jitter_hist: deque[float] = deque(maxlen=30)

        self.title = QLabel(f"{index}. {stage.title}")
        self.title.setStyleSheet("font-size: 13px; font-weight: 600; color: #ececec;")
        self.desc = QLabel(stage.description)
        self.desc.setStyleSheet("font-size: 11px; color: #95a0b2;")
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(DRAW_W // 2, DRAW_H // 2)
        self.footer = QLabel("Δ -")
        self.footer.setStyleSheet(f"font-size: 11px; color: {_COLOR_MUTE}; font-family: Menlo, Courier, monospace;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(2)
        layout.addWidget(self.title)
        layout.addWidget(self.desc)
        layout.addWidget(self.image, stretch=1)
        layout.addWidget(self.footer)
        self.setStyleSheet("background: #10131a; border: 1px solid #262c38;")
        self.render(None)

    def _project(self, pts: np.ndarray, frame_size: tuple[int, int] | None) -> np.ndarray:
        kind = self.stage.kind
        if kind == "px2d":
            if frame_size is None:
                return np.full((pts.shape[0], 2), np.nan)
            fw, fh = frame_size
            s = min(DRAW_W / fw, DRAW_H / fh)
            ox, oy = (DRAW_W - fw * s) / 2, (DRAW_H - fh * s) / 2
            return np.stack([pts[:, 0] * s + ox, pts[:, 1] * s + oy], axis=-1)
        half = _HALF_RANGE[kind]
        s = min(DRAW_W, DRAW_H) * 0.46 / half
        cx, cy = DRAW_W / 2, DRAW_H / 2
        x = pts[:, 0] * s + cx
        y = pts[:, 1] * s
        y = cy - y if _FLIP_Y[kind] else cy + y
        return np.stack([x, y], axis=-1)

    def _jitter(self, pts: np.ndarray | None) -> float | None:
        if pts is None:
            self._prev = None
            return None
        cur = pts[:, : (2 if self.stage.kind == "px2d" else 3)]
        value = None
        if self._prev is not None and self._prev.shape == cur.shape:
            d = np.linalg.norm(cur - self._prev, axis=1)
            d = d[np.isfinite(d)]
            if d.size:
                value = float(d.max())
                self._jitter_hist.append(value)
        self._prev = cur.copy()
        return value

    def render(self, snap: StageSnapshot | None) -> None:
        data = snap.data if snap is not None else None
        if data is None:
            self._jitter(None)
            self._blit(np.full((DRAW_H, DRAW_W, 3), _BG, dtype=np.uint8))
            self._set_footer(snap.note if (snap is not None and snap.note) else "데이터 없음", _COLOR_MUTE)
            return

        if self.stage.kind == "image":
            self._blit(np.asarray(data))
            self._set_footer(snap.note or "", _COLOR_MUTE)
            return

        canvas = np.full((DRAW_H, DRAW_W, 3), _BG, dtype=np.uint8)
        pts = np.asarray(data, dtype=float)
        proj = self._project(pts, snap.frame_size)
        valid = np.isfinite(proj).all(axis=1)
        if self.stage.kind == "mp3d":
            pairs, colors = POSE_CONNECTIONS, [_MP_BONE] * len(POSE_CONNECTIONS)
        else:
            pairs, colors = COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR
        for (i, j), color in zip(pairs, colors):
            if i < len(valid) and j < len(valid) and valid[i] and valid[j]:
                p1 = tuple(int(v) for v in np.round(proj[i]))
                p2 = tuple(int(v) for v in np.round(proj[j]))
                cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)
        for i in range(len(valid)):
            if valid[i]:
                cv2.circle(canvas, tuple(int(v) for v in np.round(proj[i])), 3, _JOINT, -1, cv2.LINE_AA)
        self._blit(canvas)

        jit = self._jitter(pts)
        unit = self.stage.unit
        if not self._jitter_hist:
            self._set_footer("Δ -", _COLOR_MUTE)
            return
        avg = float(np.mean(self._jitter_hist))
        color = _COLOR_BAD if avg >= _JITTER_BAD[unit] else _COLOR_WARN if avg >= _JITTER_WARN[unit] else _COLOR_OK
        fmt = "%.1f" if unit == "px" else "%.3f"
        text = f"Δmax {fmt % (jit or 0)}{unit}   avg30 {fmt % avg}{unit}"
        if snap.note:
            text += f"   {snap.note}"
        self._set_footer(text, color)

    def _set_footer(self, text: str, color: str) -> None:
        self.footer.setText(text)
        self.footer.setStyleSheet(f"font-size: 11px; color: {color}; font-family: Menlo, Courier, monospace;")

    def _blit(self, bgr: np.ndarray) -> None:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.image.setPixmap(QPixmap.fromImage(image).scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


class DebugWindow(QWidget):
    """공정 1~10을 5×2 격자로 보여준다. 워커가 flush한 최신 프레임만 20Hz로 그린다."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI Trainer · 디버깅 — 스켈레톤 처리 공정")
        self.setStyleSheet("background: #0b0e14; color: #e6e6e6;")
        self.resize(COLUMNS * (DRAW_W + 20) + 40, 2 * (DRAW_H + 80) + 70)

        self.header = QLabel("파이프라인 대기 중…")
        self.header.setStyleSheet("font-size: 13px; color: #aab4c4; padding: 4px 6px; font-family: Menlo, Courier, monospace;")

        grid = QGridLayout()
        grid.setSpacing(6)
        self.panels: dict[str, _StagePanel] = {}
        for n, stage in enumerate(STAGES, start=1):
            panel = _StagePanel(n, stage)
            self.panels[stage.id] = panel
            row, col = divmod(n - 1, COLUMNS)
            grid.addWidget(panel, row, col)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.addWidget(self.header)
        root.addLayout(grid)

        self._latest: dict[str, object] | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._paint_latest)
        self._timer.start(50)

    def on_frame(self, frame: dict[str, object]) -> None:
        self._latest = frame

    def _paint_latest(self) -> None:
        frame = self._latest
        if frame is None:
            return
        self._latest = None
        meta = frame.get("_meta") if isinstance(frame.get("_meta"), dict) else {}
        for stage_id, panel in self.panels.items():
            snap = frame.get(stage_id)
            panel.render(snap if isinstance(snap, StageSnapshot) else None)
        bits = []
        if "fps" in meta:
            bits.append(f"처리 {meta['fps']:.1f} fps")
        if "timing" in meta:
            t = meta["timing"]
            bits.append("read %.1fms · mediapipe %.1fms · 나머지 %.1fms" % (t.get("read", 0), t.get("mediapipe", 0), t.get("rest", 0)))
        if "view" in meta:
            bits.append(f"시점 {meta['view']}")
        if "filters" in meta:
            bits.append(f"필터 {'OFF' if meta['filters'] == 'off' else 'ON'}")
        if meta.get("phase"):
            bits.append(f"phase {meta['phase']}")
        if meta.get("status"):
            bits.append(str(meta["status"]))
        self.header.setText("   |   ".join(bits) if bits else "파이프라인 대기 중…")


__all__ = ["DebugWindow"]
