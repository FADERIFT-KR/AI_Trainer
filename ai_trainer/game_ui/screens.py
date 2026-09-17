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
from ai_trainer.rep_phase_display import rep_phase_label

from .pipeline_worker import PipelineStatus, SquatPipelineWorker
from .reference_track import REFERENCE_FPS, ReferenceTrack, list_available
from .analysis_dialog import AnalysisHistoryDialog
from .analysis_log import SessionAnalysisLog

REF_PANEL_W, REF_PANEL_H = 480, 480

PHASE_LABEL_KR = {"prep": "준비", "descend": "하강", "bottom": "최저", "ascend": "상승", None: "-"}

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
        self.exercise_name = "에어스쿼트"
        self.analysis_history = SessionAnalysisLog()
        self.analysis_dialog: AnalysisHistoryDialog | None = None
        self._final_match_display: float | None = None

        self.raw_camera_panel = ImagePanel("카메라 준비 중…")
        self.camera_panel = ImagePanel("스켈레톤 준비 중…")
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
        self.analysis_button = QPushButton("분석 기록")
        self.analysis_button.setToolTip("현재 세션의 REP별 자세 분석 결과 보기")
        self.analysis_button.clicked.connect(self._show_analysis_history)
        header.addWidget(self.analysis_button)
        header.addWidget(self.rep_label)
        header.addWidget(self.fps_label)

        views = QHBoxLayout()
        views.setSpacing(12)
        views.addWidget(self._panel("웹캠", self.raw_camera_panel), 1)
        views.addWidget(self._panel("실시간 스켈레톤", self.camera_panel), 1)
        views.addWidget(self._panel("동작 가이드", self.ref_panel), 1)

        self.judge_label = QLabel("대기 중")
        self.judge_label.setAlignment(Qt.AlignCenter)
        self.judge_label.setStyleSheet(
            "font-size: 20px; font-weight: 700; padding: 10px; border-radius: 8px; "
            "background: #232a38; color: #cfd6e2;"
        )

        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 14px; color: #8f9aaa;")

        self.depth_debug_label = QLabel("깊이 진단: 자세 분석 대기 중")
        self.depth_debug_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.depth_debug_label.setWordWrap(True)
        self.depth_debug_label.setStyleSheet(
            "font-family: Consolas, 'Malgun Gothic'; font-size: 12px; color: #aeb8c7; "
            "background: #171c26; padding: 5px 8px; border-radius: 5px;"
        )

        self.live_phase_label = QLabel("Live 단계: 준비 중")
        self.live_phase_label.setAlignment(Qt.AlignCenter)
        self.live_phase_label.setStyleSheet(
            "font-size: 24px; font-weight: 800; padding: 8px; border-radius: 8px; "
            "background: #1b2433; color: #7db5ff;"
        )

        self.rep_debug_label = QLabel("REP 상태: READY")
        self.rep_debug_label.setWordWrap(True)
        self.rep_debug_label.setStyleSheet(
            "font-family: Consolas, 'Malgun Gothic'; font-size: 12px; color: #f2bd61; "
            "background: #171c26; padding: 5px 8px; border-radius: 5px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addLayout(header)
        layout.addLayout(views, 1)
        layout.addWidget(self.live_phase_label)
        layout.addWidget(self.judge_label)
        layout.addWidget(self.depth_debug_label)
        layout.addWidget(self.rep_debug_label)
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
        self.ref_track = ReferenceTrack(class_label=class_label, medoid_rank=medoid_rank, tier="ground_truth")
        self._ref_tf = fit_transform(self.ref_track.coords[:, :, [0, 1]], REF_PANEL_W, REF_PANEL_H, flip_y=True)

        self._countdown_timer.stop()
        self._countdown_started = False
        self._framing_ok_since = None
        self.countdown_label.hide()
        self.result_label.setText("")
        self.analysis_history.clear()
        self._final_match_display = None
        if self.analysis_dialog is not None:
            self.analysis_dialog.close()
            self.analysis_dialog = None

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
        if self.analysis_dialog is not None:
            self.analysis_dialog.close()
            self.analysis_dialog = None
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

    def _advance_reference_panel(self) -> None:
        """REFERENCE_FPS 타이머에서만 호출됨 — 사용자 상태를 전혀 참조하지 않고
        정상 배속으로 한 프레임씩 진행(끝나면 반복)한다."""
        render_started = time.perf_counter()
        if self.ref_track is None:
            return
        ref_xy = self.ref_track.step()
        canvas = np.zeros((REF_PANEL_H, REF_PANEL_W, 3), dtype=np.uint8)
        draw_skeleton_panel(
            canvas, (0, 0), REF_PANEL_W, REF_PANEL_H, self._ref_tf(ref_xy),
            f"정상 레퍼런스 · 기준 속도 ({self.ref_track.phase_at()})", None,
            COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
        )
        self.ref_panel.set_bgr_frame(canvas)
        self._last_right_render_ms = (time.perf_counter() - render_started) * 1000.0

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

    def _show_analysis_history(self) -> None:
        print("\n========== ANALYSIS LOG TRACE ==========", flush=True)
        print(f"analysis_log total items: {len(self.analysis_history.entries)}", flush=True)
        print(f"dialog visible items: {len(self.analysis_history.entries)}", flush=True)
        print("========================================\n", flush=True)
        if self.analysis_dialog is None:
            self.analysis_dialog = AnalysisHistoryDialog(
                self.analysis_history, self.exercise_name, self
            )
        self.analysis_dialog.refresh()
        self.analysis_dialog.show()
        self.analysis_dialog.raise_()
        self.analysis_dialog.activateWindow()

    def _on_status(self, status: PipelineStatus) -> None:
        ui_started = time.perf_counter()
        self.raw_camera_panel.set_bgr_frame(status.raw_video_bgr)
        self.camera_panel.set_bgr_frame(status.live_skeleton_bgr)
        self.fps_label.setText(f"{status.fps:4.1f} FPS")
        self.rep_label.setText(f"REP {status.rep_count}")
        realtime_match = status.realtime_posture_match or {}
        state_debug = status.state_debug or {}
        if state_debug.get("event") == "rep_start":
            self._final_match_display = None
        if not status.active or not state_debug.get("calibration_ready", False):
            match_line = "자세 일치율: --"
        elif realtime_match.get("match_valid"):
            match_line = f"자세 일치율: {realtime_match['match_percent']:.0f}%"
        else:
            match_line = "자세 일치율: 측정 중"
        if self._final_match_display is not None and status.phase == "prep":
            match_line = f"최종 자세 일치율: {self._final_match_display:.0f}%"
        self.live_phase_label.setText(
            f"현재 단계: {rep_phase_label(status.phase, active=status.active)}\n{match_line}"
        )

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
        self._update_depth_debug(status.depth_debug, status.partial_distance)
        self._update_rep_debug(status.state_debug)

        # 우측 레퍼런스 패널은 이제 여기서 갱신하지 않는다 — _advance_reference_panel()이
        # 독립된 REFERENCE_FPS 타이머로 갱신한다(사용자 상태와 무관하게 정상 배속 유지).

        # 위치/화각/정면 여부가 학습 데이터(camera1) 조건에 안 맞으면 DTW 판정 대신
        # 위치 안내부터 보여준다 — 잘못된 위치에서 나온 "확인 필요"는 의미가 없다.
        if not status.framing_ok:
            self._set_judge(status.framing_message, "positioning")
        elif status.partial_distance:
            dvals = status.partial_distance["distance_by_class"]
            best_class = status.partial_distance.get("predicted_class")
            sorted_d = sorted(dvals.values())
            margin = sorted_d[1] - sorted_d[0] if len(sorted_d) >= 2 else 0.0
            if best_class is None:
                raw_best = status.partial_distance.get("raw_best_class")
                raw_hint = f" · 잠정 DTW: {raw_best}" if raw_best else ""
                self._set_judge(f"동작 분석 중{raw_hint}", "neutral")
            elif best_class == "정상" and margin > 0.05:
                self._set_judge("자세 양호", "good")
            elif best_class == "정상":
                self._set_judge("확인 필요", "neutral")
            else:
                self._set_judge(f"자세 확인 필요 ({best_class})", "bad")
        else:
            self._set_judge(status.framing_message, "neutral")

        recorded = None
        if status.completed_rep is not None:
            r = status.completed_rep
            print("\n========== ANALYSIS LOG TRACE ==========", flush=True)
            print("screens final signal received: True", flush=True)
            print(f"screens received REP: {r.rep_index + 1}", flush=True)
            print("========================================\n", flush=True)
            recorded = self.analysis_history.record_completed_rep(self.exercise_name, r)
            semantic = getattr(r, "debug_summary", None) or {}
            display_result = "자세 인식 불안정" if r.predicted_class == "자세추정불확실" else r.predicted_class
            reason_text = semantic.get("reason", "") if r.predicted_class == "자세추정불확실" else ""
            posture = getattr(r, "posture_score", None) or {}
            if posture.get("pending"):
                posture_text = "최종 자세 일치율: 측정 중"
            elif posture.get("score_valid"):
                self._final_match_display = float(posture["overall"])
                posture_text = (
                    f"최종 자세 일치율: {posture['overall']:.0f}%\n"
                    f"깊이 {posture['depth']:.0f}% · 고관절 {posture['hip']:.0f}% · "
                    f"무릎 {posture['knee']:.0f}% · 발뒤꿈치 안정 {posture['heel_stability']:.0f}% · "
                    f"좌우 균형 {posture['balance']:.0f}% · 동작 궤적 {posture['trajectory']:.0f}%"
                )
            else:
                posture_text = "최종 자세 일치율: 측정 불가"
            self.result_label.setText(
                f"REP 종료 → 판정: {display_result}\n{posture_text}\n"
                f"주요 특징: {', '.join(name for name, _ in r.top_contributing_features)}\n"
                f"2D knee/hip excursion: {semantic.get('knee_excursion_2d', 0.0):.1f}° / "
                f"{semantic.get('hip_excursion_2d', 0.0):.1f}°  |  "
                f"2D depth: {'PASS' if semantic.get('depth_2d_pass') else 'FAIL'}  |  "
                f"3D consistency: {'PASS' if semantic.get('consistency_pass') else 'FAIL'}  |  "
                f"DTW raw: {semantic.get('raw_dtw_predicted_class', '-')}"
                + (f"\n원인: {reason_text}" if reason_text else "")
            )
        else:
            recorded = self.analysis_history.observe_partial(
                self.exercise_name, status.partial_distance
            )

        if recorded is not None and self.analysis_dialog is not None and self.analysis_dialog.isVisible():
            self.analysis_dialog.refresh()
        if self.worker is not None and status.capture_monotonic:
            self.worker.record_display(
                status.camera_frame_id,
                status.capture_monotonic,
                (time.perf_counter() - ui_started) * 1000.0,
                float(getattr(self, "_last_right_render_ms", 0.0)),
            )

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"오류: {message}")
        self.status_label.setStyleSheet("color: #ff7b7b; font-weight: 600;")

    def _update_depth_debug(self, debug: dict | None, partial: dict | None = None) -> None:
        if not debug:
            self.depth_debug_label.setText("깊이 진단: 자세 분석 대기 중")
            return

        def value(name: str, digits: int = 3) -> str:
            item = debug.get(name)
            return "-" if item is None else f"{item:.{digits}f}"

        error = debug.get("hip_down_error")
        error_text = "대기" if error is None else str(bool(error))
        distances = partial.get("distance_by_class", {}) if partial else {}
        normal_dtw = distances.get("정상")
        hip_down_dtw = distances.get("엉덩이하방오류")
        dtw_difference = (
            normal_dtw - hip_down_dtw
            if normal_dtw is not None and hip_down_dtw is not None
            else None
        )
        debug_with_dtw = dict(debug)
        debug_with_dtw.update(
            {
                "normal_dtw": normal_dtw,
                "hip_down_dtw": hip_down_dtw,
                "dtw_difference": dtw_difference,
            }
        )
        debug = debug_with_dtw
        all_distances = "   ".join(
            f"{name}: {distance:.3f}"
            for name, distance in sorted(distances.items(), key=lambda item: item[1])
        ) or "-"
        raw_best = partial.get("raw_best_class") if partial else None
        validated = partial.get("validated_class") if partial else None
        sorted_values = sorted(distances.values())
        margin = sorted_values[1] - sorted_values[0] if len(sorted_values) > 1 else None
        self.depth_debug_label.setText(
            "2D 이미지(y↓)  Hip L/R/평균: "
            f"{value('left_hip_y')} / {value('right_hip_y')} / {value('hip_y')}   "
            "Knee L/R/평균: "
            f"{value('left_knee_y')} / {value('right_knee_y')} / {value('knee_y')}   "
            f"Hip-Knee: {value('hip_knee_diff')}\n"
            "3D 정규화  Pelvis 현재/REP최저: "
            f"{value('pelvis_height')} / {value('rep_min_pelvis_height')}   "
            f"Squat depth: {value('squat_depth')}   정상 기준: ≤ {value('normal_depth_threshold')}   "
            f"Hip/Knee angle: {value('hip_angle', 1)}° / {value('knee_angle', 1)}°   "
            f"엉덩이하방오류: {error_text}"
            "\n동일 프레임 2D Hip L/R/평균: "
            f"{value('left_hip_angle_2d', 1)} / {value('right_hip_angle_2d', 1)} / {value('hip_angle_2d', 1)}°   "
            "Knee L/R/평균: "
            f"{value('left_knee_angle_2d', 1)} / {value('right_knee_angle_2d', 1)} / {value('knee_angle_2d', 1)}°"
            "\n동일 프레임 3D Hip L/R/평균: "
            f"{value('left_hip_angle_3d', 1)} / {value('right_hip_angle_3d', 1)} / {value('hip_angle_3d', 1)}°   "
            "Knee L/R/평균: "
            f"{value('left_knee_angle_3d', 1)} / {value('right_knee_angle_3d', 1)} / {value('knee_angle_3d', 1)}°"
            f"\nPartial DTW  Normal: {value('normal_dtw')}   Hip-down: {value('hip_down_dtw')}   "
            f"Normal-Hip-down: {value('dtw_difference')}\n"
            f"모든 클래스: {all_distances}   RAW: {raw_best or '-'}   "
            f"Validation 후보: {validated or '보류'}   Margin: "
            f"{margin:.3f}" if margin is not None else
            f"모든 클래스: {all_distances}   RAW: {raw_best or '-'}   Validation 후보: {validated or '보류'}   Margin: -"
        )

    def _update_rep_debug(self, debug: dict | None) -> None:
        if not debug:
            self.rep_debug_label.setText("REP 상태: READY · phase 데이터 대기 중")
            return
        state_labels = {
            "prep": "READY",
            "descend": "DESCENDING",
            "bottom": "BOTTOM",
            "ascend": "ASCENDING",
        }
        current = state_labels.get(debug["current_state"], debug["current_state"])
        next_state = state_labels.get(debug["next_state"], debug["next_state"].upper())
        passed = "통과" if debug["condition_passed"] else "대기"
        self.rep_debug_label.setText(
            f"REP 상태: {current} → {next_state} ({passed}, 연속 {debug['debounce_count']}/"
            f"{debug['debounce_required']})   {debug['reason']}   "
            f"velocity={debug['pelvis_velocity']:.6f}, 기준={debug['required']}   "
            f"pelvis={debug['pelvis_height']:.4f}, 복귀기준="
            f"{debug['return_height_threshold'] if debug['return_height_threshold'] is not None else '-'}"
        )


__all__ = ["SelectionScreen", "CompareScreen"]
