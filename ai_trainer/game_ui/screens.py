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
from ai_trainer.joint_feedback import STATUS_BAD, STATUS_GOOD, STATUS_WARNING, JointScore, TRACKED_JOINTS
from ai_trainer.live_pose.window import ImagePanel
from ai_trainer.live_pose.worker import CameraConfig
from ai_trainer.render import draw_skeleton_panel, fit_transform
from ai_trainer.scoring import PASS_SCORE_THRESHOLD

from .pipeline_worker import PipelineStatus, SquatPipelineWorker
from .reference_track import DIFFICULTY_LABELS, REFERENCE_FPS, ReferenceTrack, difficulty_medoid_ranks, list_available

REF_PANEL_W, REF_PANEL_H = 480, 480

PHASE_LABEL_KR = {"prep": "준비", "descend": "하강", "bottom": "최저점", "ascend": "상승", None: "-"}

# 실시간 5단계 스테퍼(요청사항: "준비자세/하강중/최저/상승/완료 5단계로 세분화해서
# 화면에 현재 상황 띄워달라") — online_dtw.OnlineSquatSession.state는 "완료"라는
# 별도 상태가 없이 REP이 끝나자마자 바로 "prep"으로 돌아가므로, "완료"는 상태가
# 아니라 status.completed_rep이 막 도착한 순간을 붙잡아 잠깐(PHASE_COMPLETE_HOLD_MS)
# 보여주는 식으로 흉내낸다.
PHASE_STAGES = ["준비자세", "하강중", "최저", "상승", "완료"]
_PHASE_STAGE_INDEX = {"prep": 0, "descend": 1, "bottom": 2, "ascend": 3}
PHASE_COMPLETE_HOLD_MS = 900

# REP 완료 알림이 화면 중앙에 떠 있는 시간(ms). 근거 문구까지 읽을 시간을 주기 위해
# 판정 클래스만 보여주던 때(1200ms)보다 늘림.
REP_POPUP_DURATION_MS = 2000

# 한 세션(게임 한 판)에서 몇 REP를 채우면 결과 화면을 띄울지.
TARGET_REPS = 5

# RepResult.top_contributing_features(dtw_compare.py FEATURE_NAMES 코드명)를 사람이
#읽을 수 있는 한국어로 바꿔서 "왜 이 판정이 나왔는지" 근거를 보여줄 때 쓴다
# (실사용 피드백: "고관절오류 근거를 알아야 진짜인지 확인할 수 있다").
FEATURE_LABEL_KR = {
    "joint_coords_3d": "전신 좌표",
    "knee_flexion_angle": "무릎 굽힘 정도",
    "hip_flexion_angle": "고관절 굽힘 정도(허리 숙임)",
    "ankle_angle": "발목 각도",
    "torso_inclination": "상체 기울기",
    "bone_direction_vectors": "몸통/다리 방향",
    "joint_velocity": "움직임 속도",
    "pelvis_trajectory": "골반 높이/속도(스쿼트 깊이)",
    "heel_height": "뒤꿈치 들림",
    "knee_toe_alignment": "무릎-발끝 정렬",
    "left_right_asymmetry": "좌우 비대칭",
}

# 관절별 오차 막대(JointBarRow) 색상/라벨 — joint_feedback.py가 phase별 CSV 실측
# 허용오차(tolerance_deg) 대비 각도오차 비율로 판정한 status 문자열
# ("good"/"warning"/"bad")을 화면에 표시할 때 쓴다.
_JOINT_STATUS_COLOR = {STATUS_GOOD: "#72df8d", STATUS_WARNING: "#f2bd61", STATUS_BAD: "#ff7b7b"}
_JOINT_STATUS_LABEL = {STATUS_GOOD: "GOOD", STATUS_WARNING: "WARNING", STATUS_BAD: "BAD"}


def joint_detail_text(js: JointScore) -> str:
    """"몇 도까지가 합격인데 얼마나 벗어났는지"를 그대로 문장으로 만든다.

    예: "104° (합격범위 70~100°, 4° 초과)" / "72° (합격범위 60~90°, 합격)".
    합격범위 폭(js.tolerance_deg)은 phase/관절마다 다르다 — AI Hub 실측 기반
    (configs/joint_angle_tolerance.json, scripts/compute_joint_angle_tolerance.py)."""
    lo, hi = js.tolerance_range_deg
    if js.within_angle_tolerance:
        return f"{js.user_angle_deg:.0f}° (합격범위 {lo:.0f}~{hi:.0f}°, 합격)"
    over = js.angle_error_deg - js.tolerance_deg
    return f"{js.user_angle_deg:.0f}° (합격범위 {lo:.0f}~{hi:.0f}°, {over:.0f}° 초과)"


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

        # "몇 도까지가 합격인데 얼마나 벗어났는지" — 판정 근거를 숫자로 바로 보여준다
        # (실사용 피드백: 판정만 보여주지 말고 근거 각도를 같이 보여달라는 요청 대응).
        self._detail_label = QLabel("-")
        self._detail_label.setStyleSheet("font-size: 10px; color: #8f9aaa;")
        self._detail_label.setWordWrap(True)
        layout.addWidget(self._detail_label)

    def _set_bar_color(self, hex_color: str) -> None:
        self._bar.setStyleSheet(
            "QProgressBar { border: 1px solid #343c4b; border-radius: 4px; background: #1a1f29; "
            "text-align: center; color: #e7ecf4; font-size: 10px; }"
            f"QProgressBar::chunk {{ background-color: {hex_color}; border-radius: 4px; }}"
        )

    def update_score(self, js: JointScore) -> None:
        pct = int(round(js.score * 100))
        self._bar.setValue(min(100, max(0, pct)))
        self._bar.setFormat(f"{js.score * 100:.1f}%")
        self._set_bar_color(_JOINT_STATUS_COLOR.get(js.status, "#8f9aaa"))
        self._status_label.setText(_JOINT_STATUS_LABEL.get(js.status, "-"))
        self._status_label.setStyleSheet(f"font-size: 12px; font-weight: 700; color: {_JOINT_STATUS_COLOR.get(js.status, '#8f9aaa')};")
        self._detail_label.setText(joint_detail_text(js))

    def clear(self) -> None:
        self._bar.setValue(0)
        self._bar.setFormat("-")
        self._set_bar_color("#343c4b")
        self._status_label.setText("-")
        self._status_label.setStyleSheet("font-size: 12px; color: #8f9aaa;")
        self._detail_label.setText("-")

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
        self.my_skeleton_panel = ImagePanel("스켈레톤 준비 중…")
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

        phase_stepper = self._build_phase_stepper()

        joint_panel = self._build_joint_panel()

        views = QHBoxLayout()
        views.setSpacing(12)
        views.addWidget(self._panel("웹캠 · 내 자세", self.camera_panel), 1)
        views.addWidget(self._panel("내 3D 스켈레톤", self.my_skeleton_panel), 1)
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
        self.result_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.addLayout(header)
        layout.addLayout(phase_stepper)
        layout.addLayout(views, 1)
        layout.addWidget(self.judge_label)
        layout.addWidget(self.result_label)

        self.setStyleSheet(
            "QWidget { background: #11151d; color: #e7ecf4; }"
            "QGroupBox { border: 1px solid #343c4b; border-radius: 9px; margin-top: 8px; font-weight: 600; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }"
            "QPushButton { padding: 6px 12px; }"
        )

        # "N / TARGET_REPS" 큰 카운터 — 세션 내내 화면 상단에 떠 있는다(요청사항: "카운트를
        # 1/5 이렇게 크게 띄워줘"). 몇 REP까지 실제로 세어졌는지 한눈에 보여서, "5번 했는데
        # 완료화면이 안 뜬다" 같은 문제가 다시 생겨도 어디서 멈췄는지 바로 보이게 한다.
        self.rep_counter_label = QLabel("0 / 5", self)
        self.rep_counter_label.setAlignment(Qt.AlignCenter)
        self.rep_counter_label.setStyleSheet(
            "background: rgba(10,14,20,190); color: #72df8d; font-size: 42px; "
            "font-weight: 800; border-radius: 14px; border: 2px solid #343c4b;"
        )
        self.rep_counter_label.hide()

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

        # REP(한 번 앉았다 일어나기) 완료 시 화면 중앙에 잠깐 뜨는 결과 알림.
        # 하강/상승 도중 계속 바뀌는 판정 라벨이 정신없다는 피드백(실사용 확인) 대응 —
        # 동작 중엔 그대로 두고, 대신 "끝났다"는 확실한 순간을 하나 크게 짚어준다.
        self.rep_popup_label = QLabel("", self)
        self.rep_popup_label.setAlignment(Qt.AlignCenter)
        self.rep_popup_label.setWordWrap(True)
        self.rep_popup_label.hide()

        self._rep_popup_timer = QTimer(self)
        self._rep_popup_timer.setSingleShot(True)
        self._rep_popup_timer.setInterval(REP_POPUP_DURATION_MS)
        self._rep_popup_timer.timeout.connect(self.rep_popup_label.hide)

        # 우측 레퍼런스 패널 전용 타이머 — 카메라/추론 속도와 완전히 무관하게 항상
        # REFERENCE_FPS(원본 AI Hub 캡처 속도, 30fps)로만 흘러간다. 이게 "동작의 기준 속도"다.
        self._ref_playback_timer = QTimer(self)
        self._ref_playback_timer.setInterval(int(1000 / REFERENCE_FPS))
        self._ref_playback_timer.timeout.connect(self._advance_reference_panel)

        # TARGET_REPS(=5)를 채우면 뜨는 "게임 완료" 결과 화면 — 확인을 누르기 전까지는
        # 안 사라진다(요청사항). 재시도 버튼으로 같은 설정(class/난이도)으로 바로 재시작.
        self._session_reps: list = []  # 이번 세션에서 끝난 RepResult들(online_dtw.RepResult)
        self._last_class_label: str | None = None
        self._last_medoid_rank: int = 0

        self.game_result_panel = QWidget(self)
        self.game_result_panel.setStyleSheet(
            "QWidget#gameResultCard { background: rgba(15,20,28,240); border: 2px solid #4c8bff; "
            "border-radius: 20px; }"
        )
        self.game_result_panel.setObjectName("gameResultCard")
        gr_layout = QVBoxLayout(self.game_result_panel)
        gr_layout.setContentsMargins(28, 24, 28, 20)
        gr_layout.setSpacing(12)

        self.game_result_title = QLabel(f"🏁 {TARGET_REPS}회 완료!")
        self.game_result_title.setAlignment(Qt.AlignCenter)
        self.game_result_title.setStyleSheet("font-size: 28px; font-weight: 800; color: #e7ecf4;")
        gr_layout.addWidget(self.game_result_title)

        self.game_result_body = QLabel("")
        self.game_result_body.setAlignment(Qt.AlignCenter)
        self.game_result_body.setWordWrap(True)
        self.game_result_body.setStyleSheet("font-size: 13px; color: #cfd6e2;")
        gr_layout.addWidget(self.game_result_body, 1)

        gr_buttons = QHBoxLayout()
        gr_buttons.setSpacing(12)
        self.retry_btn = QPushButton("🔁 재시도")
        self.retry_btn.setMinimumHeight(44)
        self.retry_btn.setStyleSheet(
            "QPushButton { font-size: 15px; font-weight: 600; border-radius: 8px; "
            "background: #232a38; color: #cfd6e2; border: 1px solid #343c4b; }"
            "QPushButton:hover { background: #2c3444; }"
        )
        self.retry_btn.clicked.connect(self._on_retry_clicked)
        self.confirm_btn = QPushButton("확인")
        self.confirm_btn.setMinimumHeight(44)
        self.confirm_btn.setStyleSheet(
            "QPushButton { font-size: 15px; font-weight: 700; border-radius: 8px; "
            "background: #2f6feb; color: white; }"
            "QPushButton:hover { background: #4c8bff; }"
        )
        self.confirm_btn.clicked.connect(self._on_confirm_clicked)
        gr_buttons.addWidget(self.retry_btn)
        gr_buttons.addWidget(self.confirm_btn)
        gr_layout.addLayout(gr_buttons)

        self.game_result_panel.hide()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._position_rep_counter_label()
        self._position_countdown_label()
        self._position_rep_popup_label()
        self._position_game_result_panel()

    def _position_rep_counter_label(self) -> None:
        w, h = 160, 70
        self.rep_counter_label.setGeometry((self.width() - w) // 2, 12, w, h)

    def _position_countdown_label(self) -> None:
        w, h = 340, 240
        self.countdown_label.setGeometry((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _position_rep_popup_label(self) -> None:
        w, h = 480, 210
        self.rep_popup_label.setGeometry((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    def _position_game_result_panel(self) -> None:
        w, h = 700, 620
        self.game_result_panel.setGeometry((self.width() - w) // 2, (self.height() - h) // 2, w, h)

    @staticmethod
    def _panel(title: str, image: ImagePanel) -> QGroupBox:
        group = QGroupBox(title)
        v = QVBoxLayout(group)
        v.setContentsMargins(10, 16, 10, 10)
        v.addWidget(image)
        return group

    _STAGE_STYLE_INACTIVE = (
        "font-size: 14px; font-weight: 600; color: #6b7484; background: #1a1f29; "
        "border: 1px solid #343c4b; border-radius: 8px; padding: 8px 4px;"
    )
    _STAGE_STYLE_ACTIVE = (
        "font-size: 14px; font-weight: 800; color: #ffffff; background: #2f6feb; "
        "border: 1px solid #4c8bff; border-radius: 8px; padding: 8px 4px;"
    )
    _STAGE_STYLE_COMPLETE = (
        "font-size: 14px; font-weight: 800; color: #0d1a10; background: #72df8d; "
        "border: 1px solid #4ade80; border-radius: 8px; padding: 8px 4px;"
    )

    def _build_phase_stepper(self) -> QHBoxLayout:
        """준비자세->하강중->최저->상승->완료 5단계를 화면 상단에 항상 띄워, 지금 어느
        단계인지 실시간으로 강조 표시한다(요청사항)."""
        row = QHBoxLayout()
        row.setSpacing(8)
        self._phase_stage_labels: list[QLabel] = []
        for stage_name in PHASE_STAGES:
            lbl = QLabel(stage_name)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(self._STAGE_STYLE_INACTIVE)
            self._phase_stage_labels.append(lbl)
            row.addWidget(lbl, 1)

        self._phase_complete_hold_timer = QTimer(self)
        self._phase_complete_hold_timer.setSingleShot(True)
        self._phase_complete_hold_timer.setInterval(PHASE_COMPLETE_HOLD_MS)
        self._phase_complete_hold_timer.timeout.connect(lambda: self._set_phase_stage(None))
        self._phase_stage_active: int | None = None
        return row

    def _set_phase_stage(self, active_idx: int | None) -> None:
        if active_idx == self._phase_stage_active:
            return
        self._phase_stage_active = active_idx
        for i, lbl in enumerate(self._phase_stage_labels):
            if i != active_idx:
                lbl.setStyleSheet(self._STAGE_STYLE_INACTIVE)
            elif active_idx == 4:  # "완료"만 성공 느낌의 초록으로 구분
                lbl.setStyleSheet(self._STAGE_STYLE_COMPLETE)
            else:
                lbl.setStyleSheet(self._STAGE_STYLE_ACTIVE)

    def _update_phase_stepper(self, status: PipelineStatus) -> None:
        if status.completed_rep is not None:
            # REP이 막 끝난 순간 — online_dtw 상태는 이미 "prep"으로 돌아가 있지만,
            # 그대로 두면 "완료"를 사람이 볼 새도 없이 바로 "준비자세"로 넘어가 버리므로
            # 잠깐(PHASE_COMPLETE_HOLD_MS) "완료"를 붙잡아 보여준다.
            self._set_phase_stage(4)
            self._phase_complete_hold_timer.start()
            return
        if self._phase_complete_hold_timer.isActive():
            return  # "완료" 표시가 아직 안 끝났으면 phase가 바뀌어도 유지
        idx = _PHASE_STAGE_INDEX.get(status.phase)
        if idx is None:
            return  # 프레이밍/인식이 한두 프레임 흔들려 phase가 잠깐 None이 돼도(실사용
            # 확인: "최저점에서 불이 꺼짐") 스테퍼를 끄지 않고 마지막 단계를 그대로 유지 —
            # CommonSkeletonBridge가 저신뢰 관절을 freeze하는 것과 같은 패턴.
        self._set_phase_stage(idx)

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
        self._latest_joint_scores: list[JointScore] | None = None

        v.addStretch(1)
        return group

    def start(self, class_label: str, medoid_rank: int) -> None:
        # 화면에 보여주는 "정답" 레퍼런스와 DTW 점수 계산 둘 다 Ground Truth 계층(AI Hub
        # 8카메라 삼각측량 실측 3D) 사용 — 실시간 3D 소스가 자체 lifting 모델(camera1 단일뷰
        # 근사)에서 MediaPipe 자체 world_landmarks로 바뀌면서(2026-08-28), 비교 대상도 그
        # lifting 모델이 재현된 Operational 계층이 아니라 이 실측 계층으로 함께 맞췄다.
        self._last_class_label, self._last_medoid_rank = class_label, medoid_rank  # 재시도 버튼용
        self.ref_track = ReferenceTrack(class_label=class_label, medoid_rank=medoid_rank, tier="ground_truth")
        self._ref_tf = fit_transform(self.ref_track.coords[:, :, [0, 1]], REF_PANEL_W, REF_PANEL_H, flip_y=True)

        self._countdown_timer.stop()
        self._countdown_started = False
        self._framing_ok_since = None
        self._bad_streak_key = None
        self._bad_streak_count = 0
        self.countdown_label.hide()
        self._rep_popup_timer.stop()
        self.rep_popup_label.hide()
        self.game_result_panel.hide()
        self._session_reps = []
        self.result_label.setText("")
        self.overall_score_label.setText("-")
        for row in self._joint_rows.values():
            row.clear()
        self._latest_joint_scores = None
        self.rep_counter_label.setText(f"0 / {TARGET_REPS}")
        self._position_rep_counter_label()
        self.rep_counter_label.show()
        self.rep_counter_label.raise_()
        self._phase_complete_hold_timer.stop()
        self._set_phase_stage(None)

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
        self._rep_popup_timer.stop()
        self.countdown_label.hide()
        self.rep_popup_label.hide()
        self.rep_counter_label.hide()
        self._phase_complete_hold_timer.stop()
        self._set_phase_stage(None)
        self.game_result_panel.hide()
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

    def _update_my_skeleton_panel(self, status: PipelineStatus) -> None:
        """웹캠(영상+오버레이) / 정상 레퍼런스 사이에, 내 3D 자세만 크게 스켈레톤으로
        그려서 보여준다(요청사항). status.aligned_frame은 online_dtw.OnlineSquatSession이
        DTW에 쓰는 것과 동일한, Hip-center+Scale+Orientation 정규화까지 끝난 좌표라서
        레퍼런스와 같은 fit_transform(self._ref_tf)을 그대로 재사용해도 스케일이 맞는다
        — 두 스켈레톤을 나란히 놓고 비교하기도 더 쉬워진다."""
        if status.aligned_frame is None or self._ref_tf is None:
            return
        canvas = np.zeros((REF_PANEL_H, REF_PANEL_W, 3), dtype=np.uint8)
        points_px = self._ref_tf(status.aligned_frame[:, [0, 1]])
        draw_skeleton_panel(
            canvas, (0, 0), REF_PANEL_W, REF_PANEL_H, points_px,
            f"내 자세 ({PHASE_LABEL_KR.get(status.phase, '-')})", None,
            COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
        )
        self.my_skeleton_panel.set_bgr_frame(canvas)

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

    _REP_POPUP_STYLES = {
        "good": "background: rgba(20,60,35,225); color: #8bffab; border: 3px solid #4ade80;",
        "bad": "background: rgba(60,20,20,225); color: #ffb3b3; border: 3px solid #ff6b6b;",
    }

    def _show_rep_popup(self, text: str, kind: str) -> None:
        """REP(한 번 앉았다 일어나기)가 끝날 때마다 화면 중앙에 잠깐 띄우는 결과 알림.
        동작 중 계속 바뀌는 judge_label과 별개로, "끝났다"는 순간 하나를 확실히 짚어준다."""
        self.rep_popup_label.setText(text)
        self.rep_popup_label.setStyleSheet(
            "font-size: 26px; font-weight: 800; border-radius: 20px; padding: 10px; "
            + self._REP_POPUP_STYLES[kind]
        )
        self._position_rep_popup_label()
        self.rep_popup_label.show()
        self.rep_popup_label.raise_()
        self._rep_popup_timer.start()

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
            self._latest_joint_scores = status.joint_scores  # REP 완료 팝업/결과 문구에서 재사용
            scores_by_name = {js.name: js for js in status.joint_scores}
            for joint_name, row in self._joint_rows.items():
                js = scores_by_name.get(joint_name)
                if js is not None:
                    row.update_score(js)
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
        self._update_my_skeleton_panel(status)
        self._update_phase_stepper(status)
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

            # 판정 근거: DTW가 실제로 비교에 쓴 요소(코드명 -> 한국어) + 관절별 실측
            # 각도/합격범위/초과분. "왜 이 판정이 나왔는지 숫자로 보여달라"는 실사용
            # 피드백 대응 — judge_label/팝업의 클래스명만으론 검증할 방법이 없었다.
            feature_labels = [FEATURE_LABEL_KR.get(name, name) for name, _ in r.top_contributing_features]
            joint_lines = [f"{js.name} {joint_detail_text(js)}" for js in (self._latest_joint_scores or [])]
            joint_summary = "  |  ".join(joint_lines) if joint_lines else "-"

            self.result_label.setText(
                f"REP 종료 → 판정: {display_class}{confidence_note}  |  유사도: {score_text}  |  "
                f"주요 원인: {', '.join(feature_labels)}\n"
                f"관절별 근거: {joint_summary}"
            )
            # 오류 REP에 대한 구체적인 원인 문구 — 벗어난 관절이 있으면 그 관절의 실측
            # 각도/합격범위/초과분을 그대로 쓰고(가장 구체적), 없으면 DTW 주요 feature명으로
            # 대체한다. 팝업뿐 아니라 최종 결과화면(_show_game_result)에도 재사용한다
            # (요청사항: "엉덩이하방오류가 자세가 정확히 어떻게 문제됐는지 완료화면에 띄워줘").
            failing_js = [js for js in (self._latest_joint_scores or []) if not js.within_angle_tolerance]
            if display_class == "정상":
                fail_reason = ""
            elif failing_js:
                # 결과화면에 REP마다 다 나열되면 길어지니 최대 2개 관절만 구체적으로 보여준다.
                fail_reason = ", ".join(f"{js.name} {joint_detail_text(js)}" for js in failing_js[:2])
                if len(failing_js) > 2:
                    fail_reason += f" 외 {len(failing_js) - 2}개"
            else:
                fail_reason = feature_labels[0] if feature_labels else ""

            self._session_reps.append({
                "index": r.rep_index, "display_class": display_class, "score_text": score_text,
                "fail_reason": fail_reason,
            })
            self.rep_counter_label.setText(f"{len(self._session_reps)} / {TARGET_REPS}")

            if len(self._session_reps) >= TARGET_REPS:
                # TARGET_REPS(=5)를 채웠다 — 매 REP마다 뜨던 작은 팝업 대신, 확인을 누를
                # 때까지 안 사라지는 최종 결과 화면으로 넘어간다(요청사항).
                self._show_game_result()
            elif display_class == "정상":
                self._show_rep_popup(f"성공! 👍\n{score_text}", "good")
            else:
                short_reason = f"{failing_js[0].name} 등 {len(failing_js)}개 관절 벗어남" if failing_js else fail_reason
                self._show_rep_popup(f"REP {r.rep_index + 1} 완료\n{display_class}\n{short_reason}", "bad")

    def _show_game_result(self) -> None:
        """TARGET_REPS(=5)를 채운 뒤 뜨는 결과 화면. 확인 버튼을 누르기 전까지 남아있고
        (요청사항), 카메라/레퍼런스 재생은 멈춰서 화면이 계속 바뀌지 않게 한다."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            self.worker.wait(5000)
        self.worker = None
        self._ref_playback_timer.stop()
        self._rep_popup_timer.stop()
        self.rep_popup_label.hide()

        n_pass = sum(1 for rep in self._session_reps if rep["display_class"] == "정상")
        lines = []
        for rep in self._session_reps:
            line = f"REP {rep['index'] + 1}: {rep['display_class']} ({rep['score_text']})"
            if rep["fail_reason"]:
                # 오류 REP만 "자세가 정확히 어떻게 문제됐는지" 근거를 같이 보여준다(요청사항).
                line += f"\n     ↳ {rep['fail_reason']}"
            lines.append(line)
        self.game_result_body.setText(f"정상 {n_pass} / {TARGET_REPS}\n\n" + "\n".join(lines))
        self._position_game_result_panel()
        self.game_result_panel.show()
        self.game_result_panel.raise_()

    def _on_retry_clicked(self) -> None:
        self.game_result_panel.hide()
        if self._last_class_label is not None:
            self.start(self._last_class_label, self._last_medoid_rank)

    def _on_confirm_clicked(self) -> None:
        self.game_result_panel.hide()
        self._on_back()

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"오류: {message}")
        self.status_label.setStyleSheet("color: #ff7b7b; font-weight: 600;")


__all__ = ["SelectionScreen", "CompareScreen"]
