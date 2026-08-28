"""운동 선택 화면 + 좌(웹캠)/우(정상 레퍼런스) 비교 화면."""
from __future__ import annotations

import time

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import (
    QButtonGroup,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ai_trainer.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS
from ai_trainer.joint_feedback import STATUS_BAD, STATUS_GOOD, STATUS_WARNING, TRACKED_JOINTS
from ai_trainer.live_pose.window import ImagePanel
from ai_trainer.live_pose.worker import CameraConfig
from ai_trainer.render import draw_skeleton_panel, fit_transform
from ai_trainer.scoring import PASS_SCORE_THRESHOLD

from .pipeline_worker import PipelineStatus, SquatPipelineWorker
from .reference_track import DIFFICULTY_LABELS, REFERENCE_FPS, ReferenceTrack, difficulty_medoid_ranks, list_available

REF_PANEL_W, REF_PANEL_H = 480, 480

PHASE_LABEL_KR = {"prep": "준비", "descend": "하강", "bottom": "최저점", "ascend": "상승", None: "-"}

# 관절별 오차 막대(JointBarRow) 색상/라벨 — joint_feedback.py의 GREEN/YELLOW_THRESHOLD로
# 판정된 status 문자열("good"/"warning"/"bad")을 화면에 표시할 때 쓴다.
_JOINT_STATUS_COLOR = {STATUS_GOOD: "#72df8d", STATUS_WARNING: "#f2bd61", STATUS_BAD: "#ff7b7b"}
_JOINT_STATUS_LABEL = {STATUS_GOOD: "GOOD", STATUS_WARNING: "WARNING", STATUS_BAD: "BAD"}


class JointBarRow(QWidget):
    """관절 하나에 대한 "이름 + 막대바 + 오차%/판정" 한 줄.

    joint_feedback.compute_joint_scores()가 매 프레임 계산하는 JointScore(0~1 오차
    점수)를 그대로 받아 그려준다 — DTW 시퀀스 유사도(judge_label)와는 별개로,
    "지금 이 순간 이 관절이 기준 자세에서 얼마나 벗어났는지"만 보여준다.
    """

    def __init__(self, joint_name: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(2)

        header = QHBoxLayout()
        self._name_label = QLabel(joint_name)
        self._name_label.setStyleSheet("font-size: 12px; color: #cfd6e2; font-weight: 600;")
        self._status_label = QLabel("-")
        self._status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._status_label.setStyleSheet("font-size: 12px; color: #8f9aaa;")
        header.addWidget(self._name_label)
        header.addWidget(self._status_label)
        layout.addLayout(header)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setValue(0)
        self._bar.setTextVisible(True)
        self._bar.setFixedHeight(16)
        self._set_bar_color(_JOINT_STATUS_COLOR[STATUS_GOOD])
        layout.addWidget(self._bar)

    def _set_bar_color(self, hex_color: str) -> None:
        self._bar.setStyleSheet(
            "QProgressBar { border: 1px solid #343c4b; border-radius: 4px; background: #1a1f29; "
            "text-align: center; color: #e7ecf4; font-size: 10px; }"
            f"QProgressBar::chunk {{ background-color: {hex_color}; border-radius: 4px; }}"
        )

    def update_score(self, score: float, status: str) -> None:
        pct = int(round(score * 100))
        self._bar.setValue(min(100, max(0, pct)))
        self._bar.setFormat(f"{score * 100:.1f}%")
        self._set_bar_color(_JOINT_STATUS_COLOR.get(status, "#8f9aaa"))
        self._status_label.setText(_JOINT_STATUS_LABEL.get(status, "-"))
        self._status_label.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {_JOINT_STATUS_COLOR.get(status, '#8f9aaa')};")

    def clear(self) -> None:
        self._bar.setValue(0)
        self._bar.setFormat("-")
        self._set_bar_color("#343c4b")
        self._status_label.setText("-")
        self._status_label.setStyleSheet("font-size: 12px; color: #8f9aaa;")

# 화각이 이만큼(초) 연속으로 안정적으로 좋아야 3-2-1 카운트다운을 자동 시작한다.
# 순간적인 흔들림으로 바로 시작해버리는 것을 막기 위한 디바운스.
FRAMING_STABLE_SECONDS = 1.0
COUNTDOWN_START_VALUE = 3

# 1등과 2등 클래스 사이 DTW distance 차이가 이보다 작으면 "근소한 차이/애매함"으로
# 보고 특정 오류명을 확정적으로 보여주지 않는다. Offline 평가 정확도가 72.6%에
# 불과해(configs/dtw_feature_weights.json) 근소한 margin까지 특정 오류로 단정하면
# 실제로는 정상인 자세도 자꾸 오류로 잘못 표시되는 문제가 있었다(margin은 원래
# "정상"이 1등일 때만 완화에 쓰였는데, 오류 클래스가 1등일 때도 똑같이 적용해야
# 형평에 맞다).
JUDGE_MARGIN_THRESHOLD = 0.05

# 특정 오류 라벨("자세 확인 필요 (X오류)")은 같은 (phase, 오류클래스) 조합이 이 프레임 수만큼
# 연속으로 나왔을 때만 확정 표시한다 — 그 전까지는 "확인 필요"(중립)로만 보여준다.
# 순간적인 노이즈로 오류 라벨이 깜빡이며 뜨는 문제(실사용 확인) 완화용, 프레이밍 판정에
# 쓰는 FRAMING_DEBOUNCE_FRAMES 히스테리시스와 동일한 패턴.
BAD_LABEL_MIN_STREAK = 4


class SelectionScreen(QWidget):
    """1) 운동 종목을 선택하세요 (현재는 스쿼트 하나) + 2) 난이도(레퍼런스 속도) 선택."""

    start_requested = pyqtSignal(str, int)  # class_label, medoid_rank

    DEFAULT_DIFFICULTY_IDX = 1  # "보통"

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(18)

        title = QLabel("운동 종목을 선택하세요")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 26px; font-weight: 700; color: #e7ecf4;")
        layout.addWidget(title)

        try:
            self._class_label = "정상" if "정상" in {c for c, _ in list_available()} else list_available()[0][0]
        except Exception:
            self._class_label = "정상"

        # 난이도 = 우측 레퍼런스가 보여주는 시범 동작의 템포. 같은 "정상" 클래스
        # 안에서도 medoid(배우)마다 실제 하강 속도가 달라, 그 속도 분포에서
        # 느림->빠름 순으로 골라낸 medoid_rank를 난이도에 매핑한다
        # (reference_track.difficulty_medoid_ranks 참고). 판정(DTW) 로직과는
        # 무관 — 오직 우측 화면에 뭘 보여줄지에만 영향을 준다.
        try:
            self._difficulty_ranks = difficulty_medoid_ranks(self._class_label)
        except Exception:
            self._difficulty_ranks = [0]
        self._selected_difficulty_idx = min(self.DEFAULT_DIFFICULTY_IDX, len(self._difficulty_ranks) - 1)

        difficulty_title = QLabel("난이도(레퍼런스 속도)를 선택하세요")
        difficulty_title.setAlignment(Qt.AlignCenter)
        difficulty_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #cfd6e2;")
        layout.addWidget(difficulty_title)

        difficulty_row = QHBoxLayout()
        difficulty_row.setSpacing(10)
        self._difficulty_group = QButtonGroup(self)
        self._difficulty_group.setExclusive(True)
        for idx, _rank in enumerate(self._difficulty_ranks):
            label = DIFFICULTY_LABELS[idx] if idx < len(DIFFICULTY_LABELS) else f"난이도 {idx + 1}"
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setMinimumHeight(44)
            btn.setChecked(idx == self._selected_difficulty_idx)
            btn.setStyleSheet(
                "QPushButton { font-size: 15px; font-weight: 600; border-radius: 8px; "
                "background: #232a38; color: #cfd6e2; border: 1px solid #343c4b; }"
                "QPushButton:checked { background: #2f6feb; color: white; border: 1px solid #4c8bff; }"
                "QPushButton:hover { background: #2c3444; }"
            )
            btn.clicked.connect(lambda _checked, i=idx: self._on_difficulty_selected(i))
            self._difficulty_group.addButton(btn, idx)
            difficulty_row.addWidget(btn)
        layout.addLayout(difficulty_row)

        squat_btn = QPushButton("🏋️  스쿼트 (에어스쿼트)")
        squat_btn.setMinimumHeight(72)
        squat_btn.setStyleSheet(
            "QPushButton { font-size: 18px; font-weight: 600; border-radius: 10px; "
            "background: #2f6feb; color: white; }"
            "QPushButton:hover { background: #4c8bff; }"
        )
        squat_btn.clicked.connect(self._on_start_clicked)
        layout.addWidget(squat_btn)

        hint = QLabel("추후 다른 운동 종목이 추가될 예정입니다.")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #8f9aaa;")
        layout.addWidget(hint)
        layout.addStretch(1)

    def _on_difficulty_selected(self, idx: int) -> None:
        self._selected_difficulty_idx = idx

    def _on_start_clicked(self) -> None:
        medoid_rank = self._difficulty_ranks[self._selected_difficulty_idx]
        self.start_requested.emit(self._class_label, medoid_rank)


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

        joint_panel = self._build_joint_panel()

        views = QHBoxLayout()
        views.setSpacing(12)
        views.addWidget(self._panel("웹캠 · 내 자세", self.camera_panel), 1)
        views.addWidget(self._panel("정상 레퍼런스", self.ref_panel), 1)
        views.addWidget(joint_panel, 0)

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

        # 특정 오류 라벨 확정 전 히스테리시스 상태 (BAD_LABEL_MIN_STREAK 참고).
        self._bad_streak_key: tuple[str, str] | None = None
        self._bad_streak_count = 0

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

    def _build_joint_panel(self) -> QGroupBox:
        """관절별 오차 막대 패널. DTW 결과(judge_label/result_label)와는 별개로,
        "지금 이 순간" 프레임 레벨 관절 오차(joint_feedback.py)만 보여준다."""
        group = QGroupBox("관절별 자세 오차")
        group.setMinimumWidth(260)
        group.setMaximumWidth(320)
        v = QVBoxLayout(group)
        v.setContentsMargins(12, 16, 12, 10)
        v.setSpacing(10)

        overall_box = QVBoxLayout()
        overall_title = QLabel("Overall Motion Similarity")
        overall_title.setStyleSheet("font-size: 12px; color: #8f9aaa;")
        self.overall_score_label = QLabel("-")
        self.overall_score_label.setAlignment(Qt.AlignCenter)
        self.overall_score_label.setStyleSheet("font-size: 30px; font-weight: 800; color: #cfd6e2;")
        overall_box.addWidget(overall_title)
        overall_box.addWidget(self.overall_score_label)
        v.addLayout(overall_box)

        line = QLabel()
        line.setFixedHeight(1)
        line.setStyleSheet("background: #343c4b;")
        v.addWidget(line)

        # TRACKED_JOINTS(joint_feedback.py)에 정의된 관절 목록·순서를 그대로 따른다 —
        # 관절을 추가/제거하고 싶으면 joint_feedback.TRACKED_JOINTS만 고치면 여기도 같이 바뀐다.
        self._joint_rows: dict[str, JointBarRow] = {}
        for joint_name in TRACKED_JOINTS:
            row = JointBarRow(joint_name)
            self._joint_rows[joint_name] = row
            v.addWidget(row)

        v.addStretch(1)
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
        self._bad_streak_key = None
        self._bad_streak_count = 0
        self.countdown_label.hide()
        self.result_label.setText("")
        self.overall_score_label.setText("-")
        for row in self._joint_rows.values():
            row.clear()

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

    def _update_joint_panel(self, status: PipelineStatus) -> None:
        """관절별 오차 막대 + Overall Motion Similarity 갱신.

        judge_label/result_label(DTW, 시퀀스 전체 유사도)과는 완전히 별개 경로 —
        여기서 쓰는 status.live_score/joint_scores는 online_dtw.OnlineSquatSession이
        DTW 판정과 독립적으로 계산해 넘겨준 값이다(pipeline_worker.py 참고)."""
        if status.live_score is not None:
            self.overall_score_label.setText(f"{status.live_score:.0f}%")
        else:
            self.overall_score_label.setText("-")

        if status.joint_scores:
            scores_by_name = {js.name: js for js in status.joint_scores}
            for joint_name, row in self._joint_rows.items():
                js = scores_by_name.get(joint_name)
                if js is not None:
                    row.update_score(js.score, js.status)
                else:
                    row.clear()
        else:
            for row in self._joint_rows.values():
                row.clear()

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
        self._update_joint_panel(status)

        # 우측 레퍼런스 패널은 이제 여기서 갱신하지 않는다 — _advance_reference_panel()이
        # 독립된 REFERENCE_FPS 타이머로 갱신한다(사용자 상태와 무관하게 정상 배속 유지).

        # 위치/화각/정면 여부가 학습 데이터(camera1) 조건에 안 맞으면 DTW 판정 대신
        # 위치 안내부터 보여준다 — 잘못된 위치에서 나온 "확인 필요"는 의미가 없다.
        if not status.framing_ok:
            self._set_judge(status.framing_message, "positioning")
            self._bad_streak_key = None
            self._bad_streak_count = 0
        elif status.partial_distance:
            dvals = status.partial_distance["distance_by_class"]
            best_class = min(dvals, key=dvals.get)
            sorted_d = sorted(dvals.values())
            margin = sorted_d[1] - sorted_d[0] if len(sorted_d) >= 2 else 0.0
            # score_vs_normal 점수를 REP 끝날 때까지 기다리지 않고, 지금 진행 중인
            # phase 기준 실시간 유사도(%)로 함께 보여준다 — 판정 라벨만으론 애매할 때
            # "그래도 얼마나 가까운지" 감을 잡을 수 있게.
            score_suffix = f" · 유사도 {status.live_score:.0f}%" if status.live_score is not None else ""
            # "정상" CSV 레퍼런스와의 절대 거리가 충분히 가까우면(유사도 %가
            # PASS_SCORE_THRESHOLD 이상), 다른 오류 클래스가 DTW distance상 근소하게
            # 더 가깝다는 이유만으로 오류로 확정하지 않는다 — 상승 구간 등에서 자세가
            # 살짝만 틀어져도 바로 오류로 뜨던 문제(실사용 확인) 수정.
            if status.live_score is not None and status.live_score >= PASS_SCORE_THRESHOLD:
                self._set_judge(f"자세 양호{score_suffix}", "good")
                self._bad_streak_key = None
                self._bad_streak_count = 0
            # margin이 근소하면(1등과 2등이 사실상 비슷하면) 어느 쪽이 1등이든
            # "확인 필요"로 처리 — 오류 클래스라고 해서 예외를 두지 않는다.
            elif margin <= JUDGE_MARGIN_THRESHOLD:
                self._set_judge(f"확인 필요{score_suffix}", "neutral")
                self._bad_streak_key = None
                self._bad_streak_count = 0
            elif best_class == "정상":
                self._set_judge(f"자세 양호{score_suffix}", "good")
                self._bad_streak_key = None
                self._bad_streak_count = 0
            else:
                # 같은 (phase, 오류클래스)가 BAD_LABEL_MIN_STREAK 프레임 연속으로 나와야
                # 확정 오류 라벨을 띄운다 — 그 전까지는 "확인 필요"로만 보여줘서 순간적인
                # 노이즈 한 프레임이 바로 오류 말풍선으로 뜨는 걸 막는다.
                key = (status.phase, best_class)
                if key == self._bad_streak_key:
                    self._bad_streak_count += 1
                else:
                    self._bad_streak_key = key
                    self._bad_streak_count = 1
                if self._bad_streak_count >= BAD_LABEL_MIN_STREAK:
                    self._set_judge(f"자세 확인 필요 ({best_class}){score_suffix}", "bad")
                else:
                    self._set_judge(f"확인 필요{score_suffix}", "neutral")
        else:
            self._set_judge(status.framing_message, "neutral")
            self._bad_streak_key = None
            self._bad_streak_count = 0

        if status.completed_rep is not None:
            r = status.completed_rep
            # 1등과 2등 클래스의 distance가 근소하면(=애매한 판정) 그대로 확정적인
            # 판정처럼 보이지 않도록 표시해준다 (실시간 judge_label과 동일한 기준).
            sorted_d = sorted(r.raw_distance_by_class.values())
            margin = sorted_d[1] - sorted_d[0] if len(sorted_d) >= 2 else 0.0
            score_text = f"{r.score_vs_normal:.0f}%" if r.score_vs_normal is not None else "-"
            # 실시간 judge_label과 동일한 기준: "정상" 대비 유사도가 충분히 높으면
            # 다른 클래스가 근소 우세였어도 최종 판정을 "정상"으로 표시.
            if r.score_vs_normal is not None and r.score_vs_normal >= PASS_SCORE_THRESHOLD:
                display_class, confidence_note = "정상", ""
            else:
                display_class = r.predicted_class
                confidence_note = "" if margin > JUDGE_MARGIN_THRESHOLD else " (근소한 차이 — 참고용)"
            self.result_label.setText(
                f"REP 종료 → 판정: {display_class}{confidence_note}  |  유사도: {score_text}  |  "
                f"주요 특징: {', '.join(name for name, _ in r.top_contributing_features)}"
            )

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"오류: {message}")
        self.status_label.setStyleSheet("color: #ff7b7b; font-weight: 600;")


__all__ = ["SelectionScreen", "CompareScreen"]
