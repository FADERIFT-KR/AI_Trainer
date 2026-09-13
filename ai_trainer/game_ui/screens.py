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
from ai_trainer.live_pose.render import render_3d_pose
from ai_trainer.live_pose.window import ImagePanel
from ai_trainer.live_pose.worker import CameraConfig
from ai_trainer.render import draw_skeleton_panel, fit_transform
from ai_trainer.reference_levels import NORMAL_CLASS, REFERENCE_CLASSES
from ai_trainer.reference_matching import INDETERMINATE_CLASS, UNSTABLE_CLASS

from .pipeline_worker import PipelineStatus, SquatPipelineWorker
from .reference_track import REFERENCE_FPS, ReferenceTrack, create_normal_tracks, list_available

REF_PANEL_W, REF_PANEL_H = 480, 480

PHASE_LABEL_KR = {
    "prep": "준비(선자세)",
    "descend": "하강",
    "bottom": "최저점",
    "ascend": "상승",
    "complete": "준비(선자세) 완료",
    None: "-",
}

# 비정상 레퍼런스 class를 사용자가 이해할 수 있는 자세 기준으로 표시한다.
ERROR_CRITERIA = {
    REFERENCE_CLASSES[1]: "발뒤꿈치 기준",
    REFERENCE_CLASSES[2]: "엉덩이 하방 기준",
    REFERENCE_CLASSES[3]: "고관절 기준",
}
REJECTION_MESSAGES = {
    INDETERMINATE_CLASS: "오류 유형 간 차이가 작아 판정할 수 없습니다",
    UNSTABLE_CLASS: "레퍼런스와 매칭되지 않아 동작 인식이 불안정합니다",
}

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


class RotatableSkeletonPanel(ImagePanel):
    """마우스 왼쪽 드래그로 실시간 3D skeleton 시점을 회전하는 패널."""

    def __init__(self, placeholder: str, parent=None):
        super().__init__(placeholder, parent)
        self._landmarks_3d: np.ndarray | None = None
        self._yaw_deg = -24.0
        self._pitch_deg = 0.0
        self._drag_pos = None
        self.setMouseTracking(True)

    def set_landmarks(self, landmarks: np.ndarray | None) -> None:
        self._landmarks_3d = None if landmarks is None else np.asarray(landmarks).copy()
        self._render_rotated()

    def _render_rotated(self) -> None:
        if self._landmarks_3d is None:
            return
        frame = render_3d_pose(
            self._landmarks_3d,
            width=640,
            height=480,
            connections=COMMON_BONE_INDEX_PAIRS,
            yaw_deg=self._yaw_deg,
            pitch_deg=self._pitch_deg,
        )
        self.set_bgr_frame(frame)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            delta = event.pos() - self._drag_pos
            self._drag_pos = event.pos()
            self._yaw_deg = (self._yaw_deg + delta.x() * 0.6) % 360.0
            self._pitch_deg = float(np.clip(self._pitch_deg + delta.y() * 0.6, -75.0, 75.0))
            self._render_rotated()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt override)
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class CompareScreen(QWidget):
    """2)~4) 웹캠+내 스켈레톤 / 정상 레퍼런스 스켈레톤 2분할 + 실시간 동기화 + 정오 판정."""

    back_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: SquatPipelineWorker | None = None
        self.ref_tracks: dict = {}
        self._ref_tf_by_level: dict = {}
        self._error_track: ReferenceTrack | None = None
        self._error_ref_tf = None
        self._error_ref_bounds: tuple[int, int] | None = None
        self._last_error_text = ""

        self.camera_panel = ImagePanel("실시간 포즈 준비 중…")
        self.skeleton_panel = RotatableSkeletonPanel("실시간 skeleton 준비 중…")
        self.ref_panel = ImagePanel("고급 레퍼런스 준비 중…")

        self.status_label = QLabel("초기화 중…")
        self.status_label.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.rep_label = QLabel("스쿼트 0회")
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
        views.addWidget(self._panel("실시간 포즈", self.camera_panel), 1)
        views.addWidget(self._panel("실시간 포즈 skeleton (드래그하여 회전)", self.skeleton_panel), 1)
        views.addWidget(self._panel("고급 정상 레퍼런스", self.ref_panel), 1)

        self.judge_label = QLabel("대기 중")
        self.judge_label.setAlignment(Qt.AlignCenter)
        self.judge_label.setStyleSheet(
            "font-size: 20px; font-weight: 700; padding: 10px; border-radius: 8px; "
            "background: #232a38; color: #cfd6e2;"
        )

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px; color: #8f9aaa;")

        self.completion_stage_label = QLabel("")
        self.completion_stage_label.setAlignment(Qt.AlignCenter)
        self.completion_stage_label.setStyleSheet(
            "font-size: 18px; font-weight: 700; color: #4c8bff;"
        )
        self._completion_stage_serial = 0

        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet(
            "font-size: 20px; font-weight: 700; padding: 8px; border-radius: 8px; "
            "background: #4d1d1d; color: #ff7b7b;"
        )
        self.camera_error_label = QLabel("")
        self.camera_error_label.setAlignment(Qt.AlignCenter)
        self.camera_error_label.setStyleSheet(
            "font-size: 20px; font-weight: 700; padding: 8px; border-radius: 8px; "
            "background: #4d1d1d; color: #ff7b7b;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addLayout(header)
        layout.addWidget(self.error_label)
        layout.addWidget(self.camera_error_label)
        layout.addLayout(views, 1)
        layout.addWidget(self.judge_label)
        layout.addWidget(self.completion_stage_label)
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

        # 우측 레퍼런스 패널 전용 타이머 — 카메라/추론 속도와 완전히 무관하게 항상
        # REFERENCE_FPS(원본 AI Hub 캡처 속도, 30fps)로만 흘러간다. 이게 "동작의 기준 속도"다.
        self._ref_playback_timer = QTimer(self)
        self._ref_playback_timer.setInterval(int(1000 / REFERENCE_FPS))
        self._ref_playback_timer.timeout.connect(self._advance_reference_panel)

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
        # 화면에 보여주는 "정답" 레퍼런스는 정확도가 더 중요하므로 Ground Truth 계층 사용
        # (AI Hub 3d_points.csv = 카메라 8대 삼각측량 3D, camera1 단일뷰 lifting 근사가 아님).
        # DTW 점수 계산(session_active 파이프라인)은 별개로 계속 Operational 계층을 사용한다
        # (실사용자 입력도 lifting을 거치므로 그쪽과 도메인을 맞추는 게 더 정확했음, 기존 검증 결과).
        all_tracks = create_normal_tracks(tier="ground_truth")
        self.ref_tracks = {"고급": all_tracks["고급"]}
        self._ref_tf_by_level = {
            "고급": fit_transform(
                self.ref_tracks["고급"].coords[:, :, [0, 1]], REF_PANEL_W, REF_PANEL_H, flip_y=True
            )
        }

        self._countdown_timer.stop()
        self._countdown_started = False
        self._framing_ok_since = None
        self.countdown_label.hide()
        self.result_label.setText("")
        self.completion_stage_label.setText("")
        self._completion_stage_serial += 1
        self.error_label.setText("")
        self.camera_error_label.setText("")
        self._last_error_text = ""
        self._error_track = None
        self._error_ref_tf = None
        self._error_ref_bounds = None

        self.worker = SquatPipelineWorker(config=CameraConfig())
        self.worker.status_ready.connect(self._on_status)
        self.worker.status_changed.connect(self.status_label.setText)
        self.worker.fatal_error.connect(self._on_error)
        self.worker.start()

        # 레퍼런스는 사용자 상태와 무관하게 화면에 들어오는 즉시 정상 배속으로 계속 재생.
        self._ref_playback_timer.start()

    def stop(self) -> None:
        self._countdown_timer.stop()
        self._ref_playback_timer.stop()
        self.countdown_label.hide()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5000)
        self.worker = None

    # --- 3-2-1 카운트다운: 화각이 안정되면 자동 시작하고 준비 기준은 카운트다운
    # 동안 측정한다. "GO" 순간부터는 고정된 기준으로 phase/DTW만 진행한다. ---

    def _begin_countdown(self) -> None:
        if self._countdown_started:
            return
        self._countdown_started = True
        if self.worker is not None:
            self.worker.begin_countdown_calibration()
        self._countdown_value = COUNTDOWN_START_VALUE
        self._show_countdown(str(self._countdown_value))
        self._countdown_timer.start()

    def _cancel_countdown(self) -> None:
        self._countdown_timer.stop()
        self._countdown_started = False
        self._framing_ok_since = None
        self.countdown_label.hide()
        if self.worker is not None:
            self.worker.cancel_countdown_calibration()

    def _tick_countdown(self) -> None:
        self._countdown_value -= 1
        if self._countdown_value > 0:
            self._show_countdown(str(self._countdown_value))
        elif self._countdown_value == 0:
            self._show_countdown("시작!")
        else:
            if self.worker is not None and not self.worker.calibration_ready:
                # 낮은 FPS나 일시적인 준비 자세 불안정으로 기준값이 아직 부족하면
                # 측정이 완료될 때까지 운동 시작을 보류한다.
                self._show_countdown("보정 중")
                return
            self._countdown_timer.stop()
            self.countdown_label.hide()
            if self.worker is not None:
                # 새 시도가 시작되면 이전 오류 레퍼런스 재생을 중단한다.
                self._error_track = None
                self._error_ref_tf = None
                self._error_ref_bounds = None
                self._last_error_text = ""
                self.error_label.setText("")
                self.camera_error_label.setText("")
                self.worker.activate_session()  # 측정값을 고정하고 이 순간부터 phase/DTW 시작

    def _show_countdown(self, text: str) -> None:
        self._position_countdown_label()
        self.countdown_label.setText(text)
        self.countdown_label.show()
        self.countdown_label.raise_()

    def _advance_reference_panel(self) -> None:
        """REFERENCE_FPS 타이머에서만 호출됨 — 사용자 상태를 전혀 참조하지 않고
        정상 배속으로 한 프레임씩 진행(끝나면 반복)한다."""
        if not self.ref_tracks:
            return
        canvas = np.zeros((REF_PANEL_H, REF_PANEL_W, 3), dtype=np.uint8)
        level = "고급"
        track = self.ref_tracks[level]
        title = f"{level} 정상"
        transform = self._ref_tf_by_level[level]
        if self._error_track is not None:
            track = self._error_track
            level = "오동작 레퍼런스"
            title = "오동작 레퍼런스"
            transform = self._error_ref_tf
            if self._error_ref_bounds is not None and track.current_frame >= self._error_ref_bounds[1]:
                track._cursor = self._error_ref_bounds[0]
        draw_skeleton_panel(
            canvas,
            (0, 0),
            REF_PANEL_W,
            REF_PANEL_H,
            transform(track.step()),
            f"{level} 정상",
            f"기준 속도 · {track.phase_at()}",
            COMMON_BONE_INDEX_PAIRS,
            COMMON_BONE_COLORS_BGR,
        )
        self.ref_panel.set_bgr_frame(canvas)

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

    def _show_completion_stage(self, text: str) -> None:
        self.completion_stage_label.setText(text)

    def _start_completion_stages(self) -> None:
        """반복 종료 직후 하단 상태를 종료→대기→완료 순으로 표시한다."""
        self._completion_stage_serial += 1
        serial = self._completion_stage_serial
        self._show_completion_stage("스쿼트 종료")
        QTimer.singleShot(250, lambda: self._advance_completion_stage(serial, "판정 대기"))
        QTimer.singleShot(900, lambda: self._advance_completion_stage(serial, "판정 완료"))

    def _advance_completion_stage(self, serial: int, text: str) -> None:
        if serial == self._completion_stage_serial:
            self._show_completion_stage(text)

    def _on_back(self) -> None:
        self.stop()
        self.back_requested.emit()

    def _on_status(self, status: PipelineStatus) -> None:
        self.camera_panel.set_bgr_frame(status.video_bgr)
        self.skeleton_panel.set_landmarks(status.skeleton_world)
        self.fps_label.setText(f"{status.fps:4.1f} FPS")
        self.rep_label.setText(f"스쿼트 {status.rep_count}회")

        if status.pose_found and status.framing_ok:
            conf_flag = f" [인식불안 {status.n_frozen}/18]" if status.n_frozen >= 6 else ""
            self.status_label.setText(f"● 현재 시퀀스: {PHASE_LABEL_KR.get(status.phase, '-')}{conf_flag}")
            self.status_label.setStyleSheet("color: #72df8d; font-weight: 600;")
        elif status.pose_found:
            self.status_label.setText(f"⚠ 위치 조정 필요")
            self.status_label.setStyleSheet("color: #f2bd61; font-weight: 600;")
        else:
            self.status_label.setText("○ 전신 자세를 찾는 중…")
            self.status_label.setStyleSheet("color: #f2bd61; font-weight: 600;")

        self._update_countdown(status)

        # 우측 레퍼런스 패널은 이제 여기서 갱신하지 않는다 — _advance_reference_panel()이
        # 독립된 REFERENCE_FPS 타이머로 갱신한다(사용자 상태와 무관하게 정상 배속 유지).

        # 위치/화각/정면 여부가 학습 데이터(camera1) 조건에 안 맞으면 DTW 판정 대신
        # 위치 안내부터 보여준다 — 잘못된 위치에서 나온 "확인 필요"는 의미가 없다.
        if not status.framing_ok:
            self._set_judge(status.framing_message, "positioning")
        elif status.partial_distance:
            dvals = status.partial_distance["distance_by_class"]
            best_class = status.partial_distance.get("predicted_class", min(dvals, key=dvals.get))
            phase_text = PHASE_LABEL_KR.get(status.phase, "-")
            if best_class == NORMAL_CLASS:
                self._set_judge(f"{phase_text} · 정상 포즈", "good")
            elif best_class in REJECTION_MESSAGES:
                self._set_judge(REJECTION_MESSAGES[best_class], "neutral")
            else:
                criterion = ERROR_CRITERIA.get(best_class, "자세 기준")
                self._set_judge(f"{phase_text} · 비정상 포즈 · 오류 기준: {criterion}", "bad")
        else:
            self._set_judge(status.framing_message, "neutral")

        # 상단 알림은 카메라 오류/동작 오류를 분리하고, 하단에는 정상 여부만 남긴다.
        motion_error = False
        best_class = NORMAL_CLASS
        if status.partial_distance:
            dvals = status.partial_distance["distance_by_class"]
            best_class = status.partial_distance.get("predicted_class", min(dvals, key=dvals.get))
            motion_error = best_class in ERROR_CRITERIA
        recognition_rejected = best_class in REJECTION_MESSAGES
        if not status.pose_found or not status.framing_ok:
            self.error_label.setText(f"카메라 인식 오류: {status.framing_message}")
            self._set_judge("정상 포즈 판정 대기", "neutral")
        elif recognition_rejected:
            self.error_label.setText(REJECTION_MESSAGES[best_class])
            self._set_judge("자세 재시도", "neutral")
        elif motion_error:
            self.error_label.setText(f"동작 오류: {ERROR_CRITERIA.get(best_class, '자세 기준')}")
            self._set_judge("정상 포즈 판정 대기", "neutral")
        elif status.partial_distance:
            self.error_label.setText(self._last_error_text)
            self._set_judge(f"{PHASE_LABEL_KR.get(status.phase, '-')} · 정상 포즈", "good")
        else:
            self.error_label.setText(self._last_error_text)
            self._set_judge("정상 포즈 판정 대기", "neutral")

        # 오류 종류별 위치를 고정한다: 동작 오류(상단), 카메라 오류(그 아래).
        if not status.pose_found or not status.framing_ok:
            self.error_label.setText("")
            self.camera_error_label.setText(f"카메라 인식 오류: {status.framing_message}")
        elif recognition_rejected:
            self._last_error_text = REJECTION_MESSAGES[best_class]
            self.error_label.setText(self._last_error_text)
            self.camera_error_label.setText("")
        elif motion_error:
            self._last_error_text = f"동작 오류: {ERROR_CRITERIA.get(best_class, '자세 기준')}"
            self.error_label.setText(self._last_error_text)
            self.camera_error_label.setText("")
        else:
            self.error_label.setText(self._last_error_text)
            self.camera_error_label.setText("")

        if status.completed_rep is not None:
            self._start_completion_stages()
            r = status.completed_rep
            if r.predicted_class in ERROR_CRITERIA:
                try:
                    self._error_track = ReferenceTrack(
                        class_label=r.predicted_class,
                        medoid_rank=0,
                        tier="ground_truth",
                        difficulty_level=r.matched_level,
                    )
                    self._error_ref_tf = fit_transform(
                        self._error_track.coords[:, :, [0, 1]], REF_PANEL_W, REF_PANEL_H, flip_y=True
                    )
                    # 오류 유형별로 관련 phase를 시작점으로 삼아 재생한다.
                    phase_index = {REFERENCE_CLASSES[1]: 0, REFERENCE_CLASSES[2]: 2, REFERENCE_CLASSES[3]: 1}.get(
                        r.predicted_class, 0
                    )
                    self._error_ref_bounds = self._error_track.phase_bounds_by_index(phase_index)
                    self._error_track._cursor = self._error_ref_bounds[0]
                    self._last_error_text = (
                        f"동작 오류 · 오동작 레퍼런스 재생 · 오류 기준: "
                        f"{ERROR_CRITERIA.get(r.predicted_class, '자세 기준')}"
                    )
                    # 오류 반복은 무효 처리하고 다음 시도 전에 3-2-1을 다시 시작한다.
                    if self.worker is not None:
                        self.worker.session_active = False
                    self._cancel_countdown()
                    self._framing_ok_since = None
                except (OSError, ValueError, KeyError):
                    self._error_track = None
                    self._last_error_text = "동작 오류 · 오동작 레퍼런스를 불러오지 못했습니다"
            elif r.predicted_class in REJECTION_MESSAGES:
                self._error_track = None
                self._error_ref_tf = None
                self._error_ref_bounds = None
                self._last_error_text = REJECTION_MESSAGES[r.predicted_class]
                if self.worker is not None:
                    self.worker.session_active = False
                self._cancel_countdown()
                self._framing_ok_since = None
            else:
                self._error_track = None
                self._error_ref_tf = None
                self._error_ref_bounds = None
                self._last_error_text = ""
            match_rate = getattr(r, "match_rate", None)
            rate_text = "-" if match_rate is None else f"{match_rate:.1f}%"
            criterion = ""
            if r.predicted_class in ERROR_CRITERIA:
                criterion = f"  |  오류 기준: {ERROR_CRITERIA.get(r.predicted_class, '자세 기준')}"
            elif r.predicted_class in REJECTION_MESSAGES:
                criterion = f"  |  {REJECTION_MESSAGES[r.predicted_class]}"
            result_status = (
                "정상"
                if r.predicted_class == NORMAL_CLASS
                else r.predicted_class
                if r.predicted_class in REJECTION_MESSAGES
                else "비정상"
            )
            self.result_label.setText(
                f"완료 자세 · 스쿼트 {status.rep_count}회 → 정상 여부: "
                f"{result_status}  |  "
                f"정상 매칭률: {rate_text}  |  "
                f"주요 특징: {', '.join(name for name, _ in r.top_contributing_features)}{criterion}"
            )

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"오류: {message}")
        self.status_label.setStyleSheet("color: #ff7b7b; font-weight: 600;")


__all__ = ["SelectionScreen", "CompareScreen"]
