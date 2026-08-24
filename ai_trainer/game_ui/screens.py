"""운동 선택 화면 + 좌(웹캠)/우(정상 레퍼런스) 비교 화면."""
from __future__ import annotations

import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ai_trainer.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS
from ai_trainer.live_pose.window import ImagePanel
from ai_trainer.live_pose.worker import CameraConfig
from ai_trainer.render import draw_skeleton_panel, fit_transform

from .pipeline_worker import PipelineStatus, SquatPipelineWorker
from .reference_track import ReferenceTrack, list_available

REF_PANEL_W, REF_PANEL_H = 480, 480

PHASE_LABEL_KR = {"prep": "준비", "descend": "하강", "bottom": "최저점", "ascend": "상승", None: "-"}

# 화각이 이만큼(초) 연속으로 안정적으로 좋아야 3-2-1 카운트다운을 자동 시작한다.
# 순간적인 흔들림으로 바로 시작해버리는 것을 막기 위한 디바운스.
FRAMING_STABLE_SECONDS = 1.0
COUNTDOWN_START_VALUE = 3


class SelectionScreen(QWidget):
    """1) 운동 종목을 선택하세요 (현재는 스쿼트 하나)."""

    start_requested = pyqtSignal(str, int)  # class_label, medoid_rank

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(18)

        title = QLabel("운동 종목을 선택하세요")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 700; color: #e7ecf4;")
        layout.addWidget(title)

        squat_btn = QPushButton("🏋️  스쿼트 (에어스쿼트)")
        squat_btn.setMinimumHeight(72)
        squat_btn.setStyleSheet(
            "QPushButton { font-size: 18px; font-weight: 600; border-radius: 10px; "
            "background: #2f6feb; color: white; }"
            "QPushButton:hover { background: #4c8bff; }"
        )
        try:
            classes = sorted({c for c, _ in list_available()})
        except Exception:
            classes = ["정상"]
        squat_btn.clicked.connect(lambda: self.start_requested.emit("정상" if "정상" in classes else classes[0], 0))
        layout.addWidget(squat_btn)

        hint = QLabel("추후 다른 운동 종목이 추가될 예정입니다.")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #8f9aaa;")
        layout.addWidget(hint)
        layout.addStretch(1)


class CompareScreen(QWidget):
    """2)~4) 웹캠+내 스켈레톤 / 정상 레퍼런스 스켈레톤 2분할 + 실시간 동기화 + 정오 판정."""

    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: SquatPipelineWorker | None = None
        self.ref_track: ReferenceTrack | None = None

        self.camera_panel = ImagePanel("카메라 준비 중…")
        self.ref_panel = ImagePanel("레퍼런스 준비 중…")

        self.status_label = QLabel("초기화 중…")
        self.status_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.rep_label = QLabel("REP 0")
        self.rep_label.setStyleSheet("font-size: 15px; font-weight: 600; color: #72df8d;")
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        header = QHBoxLayout()
        back_btn = QPushButton("← 종목 선택으로")
        back_btn.clicked.connect(self._on_back)
        header.addWidget(back_btn)
        header.addWidget(self.status_label, 1)
        header.addWidget(self.rep_label)
        header.addWidget(self.fps_label)

        views = QHBoxLayout()
        views.setSpacing(12)
        views.addWidget(self._panel("웹캠 · 내 자세", self.camera_panel), 1)
        views.addWidget(self._panel("정상 레퍼런스", self.ref_panel), 1)

        self.judge_label = QLabel("대기 중")
        self.judge_label.setAlignment(Qt.AlignCenter)
        self.judge_label.setStyleSheet(
            "font-size: 20px; font-weight: 700; padding: 10px; border-radius: 8px; "
            "background: #232a38; color: #cfd6e2;"
        )

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px; color: #8f9aaa;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addLayout(header)
        layout.addLayout(views, 1)
        layout.addWidget(self.judge_label)
        layout.addWidget(self.result_label)

        self.setStyleSheet(
            "QWidget { background: #11151d; color: #e7ecf4; }"
            "QGroupBox { border: 1px solid #343c4b; border-radius: 9px; margin-top: 8px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }"
            "QPushButton { padding: 6px 12px; }"
        )

        # 화면 중앙에 뜨는 3-2-1 카운트다운 (일반 레이아웃에 안 넣고 위에 겹쳐 그림)
        self.countdown_label = QLabel("", self)
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet(
            "background: rgba(10,14,20,215); color: #ffffff; font-size: 150px; "
            "font-weight: 800; border-radius: 24px; border: 2px solid #4c8bff;"
        )
        self.countdown_label.hide()

        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)
        self._countdown_value = 0
        self._countdown_started = False
        self._framing_ok_since: float | None = None

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._position_countdown_label()

    def _position_countdown_label(self) -> None:
        w, h = 340, 240
        self.countdown_label.setGeometry((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    @staticmethod
    def _panel(title: str, image: ImagePanel) -> QGroupBox:
        group = QGroupBox(title)
        v = QVBoxLayout(group)
        v.setContentsMargins(10, 16, 10, 10)
        v.addWidget(image)
        return group

    def start(self, class_label: str, medoid_rank: int) -> None:
        self.ref_track = ReferenceTrack(class_label=class_label, medoid_rank=medoid_rank)
        self._ref_tf = fit_transform(self.ref_track.coords[:, :, [0, 1]], REF_PANEL_W, REF_PANEL_H, flip_y=True)

        self._countdown_timer.stop()
        self._countdown_started = False
        self._framing_ok_since = None
        self.countdown_label.hide()
        self.result_label.setText("")

        self.worker = SquatPipelineWorker(config=CameraConfig())
        self.worker.status_ready.connect(self._on_status)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.fatal_error.connect(self._on_error)
        self.worker.start()

    def stop(self) -> None:
        self._countdown_timer.stop()
        self.countdown_label.hide()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5000)
        self.worker = None

    # --- 3-2-1 카운트다운: 화각이 안정되면 자동 시작, 세션(캘리브레이션/phase/DTW)은
    # "GO" 순간부터 시작해서 시작 시점의 사용자 판단(휴먼에러)에 기대지 않게 한다. ---

    def _begin_countdown(self) -> None:
        if self._countdown_started:
            return
        self._countdown_started = True
        self._countdown_value = COUNTDOWN_START_VALUE
        self._show_countdown(str(self._countdown_value))
        self._countdown_timer.start()

    def _cancel_countdown(self) -> None:
        self._countdown_timer.stop()
        self._countdown_started = False
        self._framing_ok_since = None
        self.countdown_label.hide()

    def _tick_countdown(self) -> None:
        self._countdown_value -= 1
        if self._countdown_value > 0:
            self._show_countdown(str(self._countdown_value))
        elif self._countdown_value == 0:
            self._show_countdown("시작!")
        else:
            self._countdown_timer.stop()
            self.countdown_label.hide()
            if self.worker is not None:
                self.worker.session_active = True  # 이 순간부터 캘리브레이션/phase/DTW 시작

    def _show_countdown(self, text: str) -> None:
        self._position_countdown_label()
        self.countdown_label.setText(text)
        self.countdown_label.show()
        self.countdown_label.raise_()

    def _update_countdown(self, status: PipelineStatus) -> None:
        if self.worker is not None and self.worker.session_active:
            return  # 이미 시작됨, 더 이상 카운트다운 로직 필요 없음
        now = time.monotonic()
        if status.framing_ok:
            if self._framing_ok_since is None:
                self._framing_ok_since = now
            elif not self._countdown_started and now - self._framing_ok_since >= FRAMING_STABLE_SECONDS:
                self._begin_countdown()
        else:
            self._framing_ok_since = None
            if self._countdown_started:
                self._cancel_countdown()

    _JUDGE_STYLES = {
        "neutral": "background: #232a38; color: #cfd6e2;",
        "positioning": "background: #4d3d1d; color: #f2bd61;",
        "good": "background: #1d4d2b; color: #72df8d;",
        "bad": "background: #4d1d1d; color: #ff7b7b;",
    }

    def _set_judge(self, text: str, kind: str) -> None:
        self.judge_label.setText(text)
        self.judge_label.setStyleSheet(
            "font-size: 20px; font-weight: 700; padding: 10px; border-radius: 8px; "
            + self._JUDGE_STYLES[kind]
        )

    def _on_back(self) -> None:
        self.stop()
        self.back_requested.emit()

    def _on_status(self, status: PipelineStatus) -> None:
        self.camera_panel.set_bgr_frame(status.video_bgr)
        self.fps_label.setText(f"{status.fps:4.1f} FPS")
        self.rep_label.setText(f"REP {status.rep_count}")

        if status.pose_found and status.framing_ok:
            conf_flag = f" [인식불안 {status.n_frozen}/18]" if status.n_frozen >= 6 else ""
            self.status_label.setText(f"● 자세 감지됨 (phase: {PHASE_LABEL_KR.get(status.phase, '-')}){conf_flag}")
            self.status_label.setStyleSheet("color: #72df8d; font-weight: 600;")
        elif status.pose_found:
            self.status_label.setText(f"⚠ 위치 조정 필요")
            self.status_label.setStyleSheet("color: #f2bd61; font-weight: 600;")
        else:
            self.status_label.setText("○ 전신 자세를 찾는 중…")
            self.status_label.setStyleSheet("color: #f2bd61; font-weight: 600;")

        self._update_countdown(status)

        # 우측 패널: phase에 맞춰 레퍼런스 진행
        if self.ref_track is not None:
            ref_xy = self.ref_track.advance(status.phase if status.pose_found else None, live_fps=status.fps)
            canvas = np.zeros((REF_PANEL_H, REF_PANEL_W, 3), dtype=np.uint8)
            draw_skeleton_panel(
                canvas, (0, 0), REF_PANEL_W, REF_PANEL_H, self._ref_tf(ref_xy),
                f"정상 레퍼런스 ({self.ref_track.current_phase})", None,
                COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
            )
            self.ref_panel.set_bgr_frame(canvas)

        # 위치/화각/정면 여부가 학습 데이터(camera1) 조건에 안 맞으면 DTW 판정 대신
        # 위치 안내부터 보여준다 — 잘못된 위치에서 나온 "확인 필요"는 의미가 없다.
        if not status.framing_ok:
            self._set_judge(status.framing_message, "positioning")
        elif status.partial_distance:
            dvals = status.partial_distance["distance_by_class"]
            best_class = min(dvals, key=dvals.get)
            sorted_d = sorted(dvals.values())
            margin = sorted_d[1] - sorted_d[0] if len(sorted_d) >= 2 else 0.0
            if best_class == "정상" and margin > 0.05:
                self._set_judge("자세 양호", "good")
            elif best_class == "정상":
                self._set_judge("확인 필요", "neutral")
            else:
                self._set_judge(f"자세 확인 필요 ({best_class})", "bad")
        else:
            self._set_judge(status.framing_message, "neutral")

        if status.completed_rep is not None:
            r = status.completed_rep
            self.result_label.setText(
                f"REP 종료 → 판정: {r.predicted_class}  |  점수: {r.score_vs_normal}  |  "
                f"주요 특징: {', '.join(name for name, _ in r.top_contributing_features)}"
            )

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"오류: {message}")
        self.status_label.setStyleSheet("color: #ff7b7b; font-weight: 600;")


__all__ = ["SelectionScreen", "CompareScreen"]
