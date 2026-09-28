"""운동 선택 화면 + 좌(웹캠)/우(정상 레퍼런스) 비교 화면."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PyQt5.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QResizeEvent
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDialog,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ai_trainer.squat.depth_source import SOURCE_LABELS, SOURCE_MEDIAPIPE
from ai_trainer.squat.metric_check import (
    LabelArchive,
    MatchError,
    VIEW_CAMERAS as METRIC_VIEW_CAMERAS,
    compare_sequences,
    find_label_archives,
    find_source_root,
    resolve_ground_truth,
    scan_source_root,
    parse_source_video,
    sibling_view_videos,
    VIEW_LABEL_KO as METRIC_VIEW_LABEL_KO,
)
from ai_trainer.squat.camera_views import VIEW_FRONT, VIEW_LABEL_KO, VIEW_LEFT, VIEW_RIGHT
from ai_trainer.core.s1_capture.camera_devices import discover_camera_devices
from ai_trainer.core.s3_mapping.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS
from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.squat.joint_feedback import STATUS_BAD, STATUS_GOOD, STATUS_WARNING, JointScore, TRACKED_JOINTS
from ai_trainer.core.ui.orbit_panel import OrbitSkeletonPanel
from ai_trainer.core.ui.window import ImagePanel
from ai_trainer.core.s1_capture.camera_worker import CameraConfig
from ai_trainer.core.ui.panel_render import draw_skeleton_panel, fit_transform
from ai_trainer.squat.scoring import PASS_SCORE_THRESHOLD
from ai_trainer.squat.session_decision import decide_session, format_session_decision

from ai_trainer.squat.game_ui.pipeline_worker import PipelineStatus, SquatPipelineWorker
from ai_trainer.squat.game_ui.post_session_replay import (
    active_error_joints,
    annotate_replay_frame,
    build_replay_views,
    view_label_ko,
    view_title,
)
from ai_trainer.squat.game_ui.reference_track import DIFFICULTY_LABELS, REFERENCE_FPS, ReferenceTrack, difficulty_medoid_ranks, list_available
from ai_trainer.squat.game_ui.session_recording import SessionRecording

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

# 단안 웹캠을 정면 -> 피험자 좌측 -> 피험자 우측으로 옮겨 각 3회 측정한다.
# AI Hub 8-camera 데이터에서 자동 식별한 camera1/camera7/camera3 시점과 대응한다.
REPS_PER_VIEW = 3
VIEW_SEQUENCE = (VIEW_FRONT, VIEW_LEFT, VIEW_RIGHT)
TARGET_REPS = REPS_PER_VIEW * len(VIEW_SEQUENCE)

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
    "visible_side_track": "보이는 쪽 다리·어깨 궤적",
}

# 관절별 오차 막대(JointBarRow) 색상/라벨 — joint_feedback.py가 phase별 CSV 실측
# 허용오차(tolerance_deg) 대비 각도오차 비율로 판정한 status 문자열
# ("good"/"warning"/"bad")을 화면에 표시할 때 쓴다.
_JOINT_STATUS_COLOR = {STATUS_GOOD: "#72df8d", STATUS_WARNING: "#f2bd61", STATUS_BAD: "#ff7b7b"}
_JOINT_STATUS_LABEL = {STATUS_GOOD: "GOOD", STATUS_WARNING: "WARNING", STATUS_BAD: "BAD"}


# AI Hub 8대 카메라 중 우리가 쓰는 세 대를, 판정 규칙이 정의된 시점 이름에 대응시킨다.
# camera1(정면)/camera7(+79도)은 프로젝트의 front/left 매핑과 그대로 일치하고,
# camera2(+43도, 사선)만 대응되는 규칙이 없어 판정은 front 기준으로 돌린다
# (메트릭 대조는 좌표만 쓰므로 이 선택에 영향을 받지 않는다).
METRIC_VIEW_TO_PIPELINE = {"front": VIEW_FRONT, "oblique": VIEW_FRONT, "side": VIEW_LEFT}
METRIC_VIEW_ORDER = ("front", "oblique", "side")


class PostSessionReplayDialog(QDialog):
    """Play raw recorded views after all required measurements have finished."""

    def __init__(self, replay_views, parent=None):
        super().__init__(parent)
        self.setWindowTitle("세션 전체 영상 다시보기")
        self.setModal(True)
        self.resize(1480, 760)
        self._views = list(replay_views)
        self._view_index = -1
        self._frame_index = 0
        self._loops = 0
        self._failed_opens = 0
        self._capture = None
        self._paused = False

        self.video_panel = ImagePanel("영상 준비 중…")
        # 3D는 드래그로 돌려 볼 수 있게 한다 — 고정 사선뷰로는 가려지는 관절이 생긴다.
        self.fused_panel = OrbitSkeletonPanel("정면·좌측·우측 종합 3D 스켈레톤")
        self.fused_panel.configure(COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
                                   half_range=1.7)
        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 18px; font-weight: 800; color: #cfe3ff;")
        self.progress_label = QLabel()
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.pause_btn = QPushButton("일시정지")
        self.pause_btn.clicked.connect(self._toggle_pause)
        close_btn = QPushButton("닫기")
        close_btn.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.pause_btn)
        buttons.addWidget(close_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        views = QHBoxLayout()
        views.setSpacing(12)
        views.addWidget(self.video_panel, 1)
        views.addWidget(self.fused_panel, 1)
        layout.addLayout(views, 1)
        layout.addWidget(self.progress_label)
        layout.addLayout(buttons)
        self.setStyleSheet("QDialog { background: #11151d; color: #e7ecf4; } QPushButton { padding: 8px 16px; }")
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._open_next_view()

    def _open_next_view(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._view_index += 1
        self._frame_index = 0
        if self._view_index >= len(self._views):
            if not self._views:
                self.title_label.setText("재생할 영상이 없습니다")
                self.pause_btn.setEnabled(False)
                self._timer.stop()
                return
            # 닫을 때까지 처음 시점부터 계속 돈다 — 자세를 여러 각도에서 반복해 보려면
            # 매번 다시 열지 않아도 되어야 한다.
            self._view_index = 0
            self._loops += 1
        import cv2
        plan = self._views[self._view_index]
        self._capture = cv2.VideoCapture(str(plan.video_path))
        if not self._capture.isOpened():
            # 반복 재생이라 전부 못 여는 경우 무한 재귀가 된다 — 한 바퀴 돌 동안
            # 하나도 못 열었으면 멈춘다.
            self._failed_opens += 1
            if self._failed_opens >= len(self._views):
                self.title_label.setText("재생할 수 있는 영상이 없습니다")
                self.progress_label.setText("녹화 파일을 열지 못했습니다.")
                self.pause_btn.setEnabled(False)
                self._timer.stop()
                return
            self._open_next_view()
            return
        self._failed_opens = 0
        self.title_label.setText(view_title(plan.view))
        self._timer.setInterval(max(1, round(1000.0 / max(plan.fps, 1.0))))
        self._timer.start()

    def _next_frame(self) -> None:
        if self._paused or self._capture is None:
            return
        success, frame = self._capture.read()
        plan = self._views[self._view_index]
        if not success or self._frame_index >= len(plan.rows):
            self._open_next_view()
            return
        row = plan.rows[self._frame_index]
        self.video_panel.set_bgr_frame(annotate_replay_frame(frame, row, plan.spans))
        pose = (plan.fused_frames[self._frame_index]
                if self._frame_index < len(plan.fused_frames) else None)
        source = (plan.fusion_sources[self._frame_index]
                  if self._frame_index < len(plan.fusion_sources) else "unavailable")
        highlight = {COMMON_JOINT_NAMES.index(name)
                     for name in active_error_joints(row, plan.spans)
                     if name in COMMON_JOINT_NAMES}
        self.fused_panel.set_points(
            pose, highlight=highlight,
            footer=("FRONT+LEFT+RIGHT FUSED" if source == "phase_fused"
                    else "CURRENT VIEW ONLY"),
        )
        loop_text = f" · {self._loops + 1}회차 반복" if self._loops else ""
        self.progress_label.setText(
            f"{view_label_ko(plan.view)} {self._frame_index + 1}/{len(plan.rows)}{loop_text} · "
            "빨간색은 확정된 오류 부위 · 오른쪽 3D는 드래그로 돌려볼 수 있습니다"
        )
        self._frame_index += 1

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self.pause_btn.setText("계속 재생" if self._paused else "일시정지")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self._timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        super().closeEvent(event)


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

    # video_path가 빈 문자열이면 카메라, 아니면 그 사전 녹화 영상을 입력으로 쓴다.
    start_requested = pyqtSignal(str, int, int, str, str, str)  # + depth_source, label_archive

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

        depth_title = QLabel("3D 추정 방식을 선택하세요")
        depth_title.setAlignment(Qt.AlignCenter)
        depth_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #cfd6e2;")
        layout.addWidget(depth_title)

        self._depth_source = SOURCE_MEDIAPIPE
        depth_row = QHBoxLayout()
        depth_row.setSpacing(10)
        self._depth_group = QButtonGroup(self)
        self._depth_group.setExclusive(True)
        for idx, (name, label, hint) in enumerate(SOURCE_LABELS):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setMinimumHeight(44)
            button.setToolTip(hint)
            button.setChecked(name == self._depth_source)
            button.setStyleSheet(
                "QPushButton { font-size: 14px; font-weight: 600; border-radius: 8px; "
                "background: #232a38; color: #cfd6e2; border: 1px solid #343c4b; }"
                "QPushButton:checked { background: #6d3fb5; color: white; border: 1px solid #9b6ff0; }"
                "QPushButton:hover { background: #2c3444; }"
            )
            button.clicked.connect(
                lambda _checked, source=name: setattr(self, "_depth_source", source)
            )
            self._depth_group.addButton(button, idx)
            depth_row.addWidget(button)
        layout.addLayout(depth_row)

        self._depth_hint = QLabel(SOURCE_LABELS[0][2])
        self._depth_hint.setAlignment(Qt.AlignCenter)
        self._depth_hint.setStyleSheet("color: #8f9aaa; font-size: 12px;")
        layout.addWidget(self._depth_hint)
        self._depth_group.buttonClicked.connect(
            lambda btn: self._depth_hint.setText(
                next(h for _n, l, h in SOURCE_LABELS if l == btn.text())
            )
        )

        source_title = QLabel("입력 소스를 선택하세요")
        source_title.setAlignment(Qt.AlignCenter)
        source_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #cfd6e2;")
        layout.addWidget(source_title)

        self._video_path = ""
        self._metric_clip = None  # AI Hub 원천데이터일 때만 채워진다
        source_row = QHBoxLayout()
        source_row.setSpacing(10)
        self._camera_source_btn = QPushButton("📷  카메라")
        self._video_source_btn = QPushButton("🎞  녹화 영상 선택…")
        for button in (self._camera_source_btn, self._video_source_btn):
            button.setCheckable(True)
            button.setMinimumHeight(44)
            button.setStyleSheet(
                "QPushButton { font-size: 15px; font-weight: 600; border-radius: 8px; "
                "background: #232a38; color: #cfd6e2; border: 1px solid #343c4b; }"
                "QPushButton:checked { background: #2f6feb; color: white; border: 1px solid #4c8bff; }"
                "QPushButton:hover { background: #2c3444; }"
            )
            source_row.addWidget(button)
        self._camera_source_btn.setChecked(True)
        self._camera_source_btn.clicked.connect(self._on_camera_source_selected)
        self._video_source_btn.clicked.connect(self._on_pick_video)
        layout.addLayout(source_row)

        # 사람이 눈으로 크로스체크할 수 있도록 원천/라벨 경로를 각각 그대로 보여준다.
        self._video_label = QLabel("")
        self._label_path_label = QLabel("")
        for widget, color in ((self._video_label, "#8bffab"), (self._label_path_label, "#9b8bff")):
            widget.setAlignment(Qt.AlignCenter)
            widget.setWordWrap(True)
            widget.setStyleSheet(
                f"color: {color}; font-size: 11px; font-family: Menlo, Courier, monospace;")
            widget.hide()
            layout.addWidget(widget)

        # 외장 하드가 붙어 있으면 클래스/난이도/배우·트라이를 드롭다운으로 바로 고른다.
        self._catalog: dict = {}
        self._catalog_root = None
        self._quick_box = QWidget()
        quick = QHBoxLayout(self._quick_box)
        quick.setContentsMargins(0, 0, 0, 0)
        quick.setSpacing(8)
        self._combo_group = QComboBox()
        self._combo_take = QComboBox()
        self._combo_view = QComboBox()
        for combo, width in ((self._combo_group, 190), (self._combo_take, 210), (self._combo_view, 110)):
            combo.setMinimumHeight(36)
            combo.setMinimumWidth(width)
            combo.setStyleSheet(
                "QComboBox { font-size: 13px; padding: 4px 8px; border-radius: 6px; "
                "background: #232a38; color: #cfd6e2; border: 1px solid #343c4b; }")
        quick.addWidget(QLabel("클래스·난이도"))
        quick.addWidget(self._combo_group, 1)
        quick.addWidget(QLabel("배우·트라이"))
        quick.addWidget(self._combo_take, 1)
        quick.addWidget(QLabel("시점"))
        quick.addWidget(self._combo_view)
        self._quick_box.hide()
        layout.addWidget(self._quick_box)

        self._combo_group.currentIndexChanged.connect(self._on_group_changed)
        self._combo_take.currentIndexChanged.connect(self._on_take_changed)
        self._combo_view.currentIndexChanged.connect(self._on_take_changed)

        self._rescan_btn = QPushButton("🔌  외장 하드에서 불러오기 (My Passport)")
        self._rescan_btn.setMinimumHeight(36)
        self._rescan_btn.setToolTip(
            "연결된 외장 하드의 에어스쿼트 폴더를 훑어 배우·트라이 목록을 채운다.")
        self._rescan_btn.clicked.connect(self._on_scan_source_root)
        layout.addWidget(self._rescan_btn)

        self._label_archive_path = ""
        self._label_candidates = list(find_label_archives())
        self._pick_label_btn = QPushButton("📑  라벨링 데이터 직접 지정…")
        self._pick_label_btn.setMinimumHeight(36)
        self._pick_label_btn.setToolTip(
            "3D 정답 CSV가 든 AI Hub 라벨링 ZIP. 영상과 1:1 대조해 좌표 매칭율을 계산한다.")
        self._pick_label_btn.clicked.connect(self._on_pick_label_archive)
        self._pick_label_btn.hide()
        layout.addWidget(self._pick_label_btn)

        self._camera_block = QWidget()
        camera_block_layout = QVBoxLayout(self._camera_block)
        camera_block_layout.setContentsMargins(0, 0, 0, 0)
        camera_block_layout.setSpacing(10)
        layout.addWidget(self._camera_block)
        layout = camera_block_layout  # 아래 카메라 관련 위젯은 이 블록 안에 넣는다

        camera_title = QLabel("사용할 카메라 장치를 선택하세요")
        camera_title.setAlignment(Qt.AlignCenter)
        camera_title.setStyleSheet("font-size: 15px; font-weight: 600; color: #cfd6e2;")
        layout.addWidget(camera_title)

        self._camera_group = QButtonGroup(self)
        self._camera_group.setExclusive(True)
        self._camera_buttons = QWidget()
        self._camera_row = QHBoxLayout(self._camera_buttons)
        self._camera_row.setContentsMargins(0, 0, 0, 0)
        self._camera_row.setSpacing(10)
        self._selected_camera_index = 0
        layout.addWidget(self._camera_buttons)
        self._populate_camera_buttons()

        refresh_camera_btn = QPushButton("카메라 목록 새로고침")
        refresh_camera_btn.setMinimumHeight(36)
        refresh_camera_btn.clicked.connect(self._populate_camera_buttons)
        layout.addWidget(refresh_camera_btn)
        layout = self.layout()  # 카메라 블록 종료 — 다시 화면 전체 레이아웃으로

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

    def _on_camera_source_selected(self) -> None:
        self._video_path = ""
        self._camera_source_btn.setChecked(True)
        self._video_source_btn.setChecked(False)
        self._video_label.hide()
        self._label_path_label.hide()
        self._pick_label_btn.hide()
        self._quick_box.hide()
        self._camera_block.show()

    _EXTERNAL_ROOT = Path("/Volumes/My Passport/에어스쿼트")

    def _show_match_error(self, error: MatchError) -> None:
        """무엇이 어긋났는지 알려주고, 해당 폴더를 바로 열 수 있게 한다."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("파일 매칭 오류")
        box.setText("영상과 라벨링 데이터를 짝지을 수 없습니다.")
        box.setInformativeText(str(error))
        open_btn = None
        if error.folder is not None and Path(error.folder).exists():
            open_btn = box.addButton("폴더 열기", QMessageBox.ActionRole)
        box.addButton("확인", QMessageBox.AcceptRole)
        box.exec_()
        if open_btn is not None and box.clickedButton() is open_btn:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(error.folder)))

    def _on_scan_source_root(self) -> None:
        """외장 하드(또는 배우 폴더만 복사해 온 경로)를 훑어 드롭다운을 채운다."""
        root = find_source_root()
        if root is None:
            root_str = QFileDialog.getExistingDirectory(
                self, "원천데이터 폴더 선택 (에어스쿼트)", str(Path.home()))
            if not root_str:
                return
            root = Path(root_str)
        try:
            self._catalog = scan_source_root(root)
        except MatchError as error:
            self._show_match_error(error)
            return
        self._catalog_root = root
        self._combo_group.blockSignals(True)
        self._combo_group.clear()
        for error_type, level in sorted(self._catalog):
            self._combo_group.addItem(
                f"{error_type} · {level}  ({len(self._catalog[(error_type, level)])}개)",
                (error_type, level))
        self._combo_group.blockSignals(False)
        self._quick_box.show()
        self._rescan_btn.setText(f"🔌  다시 훑기 · {root}")
        self._on_group_changed()

    def _on_group_changed(self) -> None:
        key = self._combo_group.currentData()
        self._combo_take.blockSignals(True)
        self._combo_take.clear()
        for entry in self._catalog.get(key, []):
            self._combo_take.addItem(entry.label, entry)
        self._combo_take.blockSignals(False)
        self._on_take_changed()

    def _on_take_changed(self) -> None:
        entry = self._combo_take.currentData()
        if entry is None:
            return
        wanted = self._combo_view.currentData()
        self._combo_view.blockSignals(True)
        self._combo_view.clear()
        for view in ("front", "oblique", "side"):
            if view in entry.videos:
                self._combo_view.addItem(METRIC_VIEW_LABEL_KO[view], view)
        if wanted is not None:
            index = self._combo_view.findData(wanted)
            if index >= 0:
                self._combo_view.setCurrentIndex(index)
        self._combo_view.blockSignals(False)
        view = self._combo_view.currentData()
        if view is None:
            return
        self._video_path = str(entry.videos[view])
        self._camera_source_btn.setChecked(False)
        self._video_source_btn.setChecked(True)
        self._camera_block.hide()
        self._refresh_match_labels()

    def _on_pick_label_archive(self) -> None:
        start = self._label_archive_path or str(Path.home() / "Project")
        path, _ = QFileDialog.getOpenFileName(
            self, "라벨링 데이터 선택 (TL.zip / VL.zip)", start,
            "AI Hub 라벨링 (*.zip);;모든 파일 (*)")
        if not path:
            return
        self._label_archive_path = path
        self._refresh_match_labels()

    def _on_pick_video(self) -> None:
        start = (str(self._EXTERNAL_ROOT) if self._EXTERNAL_ROOT.is_dir()
                 else str(Path.home() / "Desktop"))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "분석할 스쿼트 영상 선택 (AI Hub 원천데이터 또는 직접 촬영 영상)",
            start,
            "동영상 (*.avi *.mp4 *.mov *.mkv *.m4v);;모든 파일 (*)",
        )
        if not path:
            # 취소하면 직전 선택을 유지한다.
            self._video_source_btn.setChecked(bool(self._video_path))
            return
        self._video_path = path
        self._camera_source_btn.setChecked(False)
        self._video_source_btn.setChecked(True)
        self._camera_block.hide()
        self._refresh_match_labels()

    def _refresh_match_labels(self) -> None:
        """선택한 영상에서 AI Hub 구조를 읽어 대조용 라벨을 자동으로 찾아 보여준다."""
        path = Path(self._video_path)
        self._video_label.setText(f"원천(입력) · {path}")
        self._video_label.show()
        self._pick_label_btn.show()
        if self._catalog:
            self._quick_box.show()
        self._metric_clip = None

        try:
            clip = parse_source_video(path)
        except MatchError:
            # AI Hub 구조가 아니면 그냥 일반 녹화 영상으로 쓴다 — 대조는 하지 않는다.
            self._label_path_label.setText(
                "라벨(대조) · AI Hub 원천데이터 구조가 아니라 메트릭 검증 없이 재생만 합니다.")
            self._label_path_label.show()
            return

        siblings = sibling_view_videos(clip)
        views = " / ".join(METRIC_VIEW_LABEL_KO[v] for v in siblings) or "없음"
        # 한 배우/트라이는 TL(Training) 또는 VL(Validation) 한쪽에만 있다 — 실제로 가진
        # 쪽을 자동으로 고른다. 사용자가 직접 지정했으면 그것을 먼저 본다.
        candidates = ([self._label_archive_path] if self._label_archive_path else [])
        candidates += [str(c) for c in self._label_candidates]
        archive = resolve_ground_truth(clip, candidates)
        if archive is None:
            hint = ("라벨링 ZIP(TL.zip / VL.zip)을 찾지 못했습니다. 직접 지정해 주세요."
                    if not candidates else
                    f"찾은 라벨링 ZIP 어디에도 이 정답이 없습니다:\n{clip.ground_truth_3d}")
            self._label_path_label.setText(
                f"{clip.describe()} · 같은 rep 시점: {views}\n라벨(대조) · {hint}")
            self._label_path_label.show()
            self._metric_clip = clip
            return
        self._label_archive_path = str(archive.path)
        self._metric_clip = clip
        self._label_path_label.setText(
            f"{clip.describe()} · 같은 rep 시점: {views}\n"
            f"라벨(대조) · {self._label_archive_path}\n"
            f"           {clip.ground_truth_3d}")
        self._label_path_label.show()

    def _populate_camera_buttons(self) -> None:
        previous = self._selected_camera_index
        for button in self._camera_group.buttons():
            self._camera_group.removeButton(button)
        while self._camera_row.count():
            item = self._camera_row.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        devices = discover_camera_devices()
        indexes = {device.index for device in devices}
        self._selected_camera_index = previous if previous in indexes else devices[0].index
        for device in devices:
            button = QPushButton(device.button_label)
            button.setCheckable(True)
            button.setMinimumHeight(44)
            button.setChecked(device.index == self._selected_camera_index)
            button.setToolTip(device.device_name or device.description)
            button.setStyleSheet(
                "QPushButton { font-size: 14px; border-radius: 8px; background: #232a38; "
                "color: #cfd6e2; border: 1px solid #343c4b; padding: 7px; }"
                "QPushButton:checked { background: #1d4d2b; color: #8bffab; border: 1px solid #4ade80; }"
                "QPushButton:hover { background: #2c3444; }"
            )
            button.clicked.connect(
                lambda _checked, camera_index=device.index: setattr(
                    self, "_selected_camera_index", camera_index
                )
            )
            self._camera_group.addButton(button, device.index)
            self._camera_row.addWidget(button)

    def _on_start_clicked(self) -> None:
        medoid_rank = self._difficulty_ranks[self._selected_difficulty_idx]
        self.start_requested.emit(
            self._class_label, medoid_rank, self._selected_camera_index,
            self._video_path, self._depth_source, self._label_archive_path
        )


class CompareScreen(QWidget):
    """2)~4) 웹캠+내 스켈레톤 / 정상 레퍼런스 스켈레톤 2분할 + 실시간 동기화 + 정오 판정."""

    back_requested = pyqtSignal()

    def __init__(self, parent=None, debug_tap=None):
        super().__init__(parent)
        self.worker: SquatPipelineWorker | None = None
        # 디버깅 모드에서만 주어진다(ai_trainer.debug.StageTap). 워커로 그대로 넘긴다.
        self._debug_tap = debug_tap
        self._video_path = ""  # 비어 있으면 카메라 입력
        self._depth_source = SOURCE_MEDIAPIPE
        self._label_archive_path = ""
        self._metric_pred: list = []   # 이번 영상에서 나온 (T,18,3) 추정 좌표
        self._metric_reports: list = []       # 시점별 MatchReport
        self._video_view_queue: list = []     # [(metric_view, 영상경로), ...] 남은 시점
        self._metric_view = ""                # 지금 돌고 있는 AI Hub 시점 이름
        self.ref_track: ReferenceTrack | None = None

        self.camera_panel = ImagePanel("카메라 준비 중…")
        # 정면/사선/측면을 따로 띄우는 대신 한 화면에서 마우스로 돌려 본다.
        self.my_skeleton_panel = OrbitSkeletonPanel("스켈레톤 준비 중…")
        self.my_skeleton_panel.configure(
            COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
            half_range=1.7,  # 정규화 좌표(다리길이=1) 기준 전신이 담기는 반경
        )
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
        views.addWidget(
            self._panel("내 3D 추정 스켈레톤 · 드래그로 회전", self.my_skeleton_panel), 1
        )
        views.addWidget(self._panel("정상 레퍼런스", self.ref_panel), 1)
        views.addWidget(joint_panel, 0)

        # 메트릭 검증(원천 영상 <-> 라벨 정답 대조) 결과. 일반 재생에서는 숨어 있다.
        self.metric_label = QLabel("")
        self.metric_label.setAlignment(Qt.AlignCenter)
        self.metric_label.setWordWrap(True)
        self.metric_label.setStyleSheet(
            "font-size: 13px; padding: 8px; border-radius: 8px; background: #1b2434; "
            "color: #cfe3ff; font-family: Menlo, Courier, monospace;")
        self.metric_label.hide()

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
        layout.addWidget(self.metric_label)
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
        self.rep_counter_label = QLabel(f"0 / {TARGET_REPS}", self)
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

        # QWidget이 닫힌 뒤에도 QTimer.singleShot 콜백이 남아 카메라를 다시 여는 일을
        # 막기 위해 취소 가능한 소유 타이머로 시점 전환을 관리한다.
        self._view_transition_timer = QTimer(self)
        self._view_transition_timer.setSingleShot(True)
        self._view_transition_timer.setInterval(REP_POPUP_DURATION_MS)
        self._view_transition_timer.timeout.connect(self._advance_view)

        # 우측 레퍼런스 패널 전용 타이머 — 카메라/추론 속도와 완전히 무관하게 항상
        # REFERENCE_FPS(원본 AI Hub 캡처 속도, 30fps)로만 흘러간다. 이게 "동작의 기준 속도"다.
        self._ref_playback_timer = QTimer(self)
        self._ref_playback_timer.setInterval(int(1000 / REFERENCE_FPS))
        self._ref_playback_timer.timeout.connect(self._advance_reference_panel)

        # 세 시점의 TARGET_REPS를 채우면 뜨는 "게임 완료" 결과 화면 — 확인을 누르기 전까지는
        # 안 사라진다(요청사항). 재시도 버튼으로 같은 설정(class/난이도)으로 바로 재시작.
        self._session_reps: list = []  # 이번 세션에서 끝난 RepResult들(online_dtw.RepResult)
        self._view_index = 0
        self._current_view = VIEW_SEQUENCE[0]
        self._transitioning_view = False
        self._last_class_label: str | None = None
        self._last_medoid_rank: int = 0
        self._last_camera_index: int = 0
        self._session_recording: SessionRecording | None = None
        self._session_result_text = ""
        self._pending_error_replay = False
        self._error_replay_shown = False

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

        self.recording_result_label = QLabel("결과 확인 후 녹화 영상과 분석자료를 저장할 수 있습니다.")
        self.recording_result_label.setAlignment(Qt.AlignCenter)
        self.recording_result_label.setWordWrap(True)
        self.recording_result_label.setStyleSheet("font-size: 12px; color: #9fc6ff;")
        gr_layout.addWidget(self.recording_result_label)

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
        self.save_recording_btn = QPushButton("녹화 영상·분석자료 저장…")
        self.save_recording_btn.setMinimumHeight(44)
        self.save_recording_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: 700; border-radius: 8px; "
            "background: #1d4d2b; color: #b9f6c8; }"
            "QPushButton:hover { background: #266838; }"
        )
        self.save_recording_btn.clicked.connect(self._save_recording)
        self.replay_error_btn = QPushButton("🎬  세션 전체 영상 다시보기")
        self.replay_error_btn.setMinimumHeight(44)
        self.replay_error_btn.setEnabled(False)
        self.replay_error_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: 700; border-radius: 8px; "
            "background: #23324a; color: #cfe3ff; border: 1px solid #4c8bff; }"
            "QPushButton:hover:enabled { background: #2d4260; }"
        )
        self.replay_error_btn.clicked.connect(self._show_recorded_error_replay)
        self.confirm_btn = QPushButton("확인")
        self.confirm_btn.setMinimumHeight(44)
        self.confirm_btn.setStyleSheet(
            "QPushButton { font-size: 15px; font-weight: 700; border-radius: 8px; "
            "background: #2f6feb; color: white; }"
            "QPushButton:hover { background: #4c8bff; }"
        )
        self.confirm_btn.clicked.connect(self._on_confirm_clicked)
        gr_buttons.addWidget(self.retry_btn)
        gr_buttons.addWidget(self.replay_error_btn)
        gr_buttons.addWidget(self.save_recording_btn)
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
    def _panel(title: str, image) -> QGroupBox:
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

    def start(self, class_label: str, medoid_rank: int, camera_index: int = 0,
              video_path: str = "", depth_source: str = SOURCE_MEDIAPIPE,
              label_archive: str = "") -> None:
        # 화면에 보여주는 "정답" 레퍼런스와 DTW 점수 계산 둘 다 Ground Truth 계층(AI Hub
        # 8카메라 삼각측량 실측 3D) 사용 — 실시간 3D 소스가 자체 lifting 모델(camera1 단일뷰
        # 근사)에서 MediaPipe 자체 world_landmarks로 바뀌면서(2026-08-28), 비교 대상도 그
        # lifting 모델이 재현된 Operational 계층이 아니라 이 실측 계층으로 함께 맞췄다.
        self._last_class_label, self._last_medoid_rank = class_label, medoid_rank  # 재시도 버튼용
        self._last_camera_index = camera_index
        # 사전 녹화 영상 모드: 한 파일에 한 시점만 담겨 있으므로 3시점 순회 없이
        # 정면 한 시점만 돌리고, 영상이 끝나면 그 시점까지의 결과로 마무리한다.
        self._video_path = video_path
        self._depth_source = depth_source
        self._label_archive_path = label_archive
        self._metric_pred = []
        self._metric_reports = []
        self._metric_view = ""
        # AI Hub 원천데이터면 같은 rep의 정면/사선/측면을 한 번에 순차로 돌린다.
        self._video_view_queue = []
        if video_path:
            try:
                clip = parse_source_video(Path(video_path))
            except MatchError:
                clip = None
            if clip is not None:
                siblings = sibling_view_videos(clip)
                self._video_view_queue = [
                    (view, siblings[view]) for view in METRIC_VIEW_ORDER if view in siblings
                ]
        self.metric_label.hide()
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            if not self.worker.wait(5000):
                self._on_error("이전 카메라 작업이 종료되지 않아 재시작할 수 없습니다.")
                return
        self.worker = None
        if self._session_recording is not None:
            self._session_recording.close()
        self._session_recording = SessionRecording()
        self._session_result_text = ""
        self._pending_error_replay = False
        self._error_replay_shown = False
        self.ref_track = ReferenceTrack(class_label=class_label, medoid_rank=medoid_rank, tier="ground_truth")
        self._ref_tf = None

        self._countdown_timer.stop()
        self._countdown_started = False
        self._framing_ok_since = None
        self._bad_streak_key = None
        self._bad_streak_count = 0
        self.countdown_label.hide()
        self._rep_popup_timer.stop()
        self._view_transition_timer.stop()
        self.rep_popup_label.hide()
        self.game_result_panel.hide()
        self.replay_error_btn.setEnabled(False)
        self.game_result_title.setText(f"🏁 {TARGET_REPS}회 완료!")
        self.recording_result_label.setText("결과 확인 후 녹화 영상과 분석자료를 저장할 수 있습니다.")
        self._session_reps = []
        self._view_index = 0
        self._current_view = VIEW_SEQUENCE[0]
        self._transitioning_view = False
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

        # AI Hub 원천데이터면 정면->사선->측면 세 영상을 한 세션에서 이어서 돌린다.
        if not self._start_next_video_view():
            self._start_view_worker(self._current_view)

        # 레퍼런스는 사용자 상태와 무관하게 화면에 들어오는 즉시 정상 배속으로 계속 재생.
        self._ref_playback_timer.start()

    @staticmethod
    def _project_view(coords_3d: np.ndarray, view: str) -> np.ndarray:
        """정규화된 3D 좌표를 선택한 촬영 시점의 화면 평면으로 투영한다."""
        if view == VIEW_FRONT:
            return coords_3d[..., [0, 1]]
        if view in (VIEW_LEFT, VIEW_RIGHT):
            projected = coords_3d[..., [2, 1]].copy()
            if view == VIEW_RIGHT:
                projected[..., 0] *= -1
            return projected
        raise ValueError(f"지원하지 않는 시점입니다: {view}")

    def _set_reference_view(self, view: str) -> None:
        if self.ref_track is None:
            return
        projected = self._project_view(self.ref_track.coords, view)
        self._ref_tf = fit_transform(projected, REF_PANEL_W, REF_PANEL_H, flip_y=True)
        self.ref_track.reset()

    def _start_view_worker(self, view: str) -> None:
        """한 물리적 촬영 시점의 3회 측정을 새 세션으로 시작한다."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            if not self.worker.wait(5000):
                self._on_error("이전 시점의 카메라 작업이 종료되지 않았습니다. 다시 시도해 주세요.")
                return
        self.worker = None
        self._current_view = view
        self._transitioning_view = False
        self._countdown_timer.stop()
        self._countdown_started = False
        self._framing_ok_since = None
        self.countdown_label.hide()
        self._phase_complete_hold_timer.stop()
        self._set_phase_stage(None)
        self.overall_score_label.setText("-")
        for row in self._joint_rows.values():
            row.clear()
        self._latest_joint_scores = None
        self._set_reference_view(view)

        # 사선(camera2)처럼 판정 규칙이 없는 각도는 view_mode를 front로 돌리므로,
        # 화면에는 실제 촬영 시점 이름을 쓴다.
        label = view_label_ko(self._metric_view) if self._metric_view else VIEW_LABEL_KO[view]
        self.status_label.setText(
            f"{label} · {Path(self._video_path).name} 분석 중…" if self._video_path
            else f"{label} 측정을 준비하세요 · 전신이 보이면 3초 뒤 시작합니다."
        )
        self.rep_label.setText(f"{label} REP 0 / {REPS_PER_VIEW}")
        self.rep_counter_label.setText(
            f"{label} 0/{REPS_PER_VIEW} · 전체 {len(self._session_reps)}/{TARGET_REPS}"
        )
        # 좌/우 관절이 화면 미러링으로 뒤바뀌지 않게 분석과 표시 모두 센서 원본을 쓴다.
        self.worker = SquatPipelineWorker(
            config=CameraConfig(
                camera_index=self._last_camera_index,
                mirror=False,
                video_path=self._video_path or None,
            ),
            view_mode=view,
            recording_dir=self._session_recording.directory if self._session_recording is not None else None,
            debug_tap=self._debug_tap,
            depth_source=self._depth_source,
            record_view=self._metric_view or view,
        )
        self.worker.status_ready.connect(self._on_status)
        self.worker.status_changed.connect(
            lambda message, source=view: self._on_worker_message(source, message)
        )
        self.worker.source_finished.connect(self._on_source_finished)
        self.worker.fatal_error.connect(
            lambda message, source=view: self._on_worker_error(source, message)
        )
        self.worker.start()

    def _start_next_video_view(self) -> bool:
        """큐에 남은 AI Hub 시점이 있으면 그 영상으로 다음 측정을 시작한다."""
        if not self._video_view_queue:
            return False
        metric_view, path = self._video_view_queue.pop(0)
        self._metric_view = metric_view
        self._video_path = str(path)
        self._metric_pred = []
        # 보는 각도(방위/고도/배율)는 시점이 바뀌어도 사용자가 맞춰 둔 그대로 유지한다.
        self._start_view_worker(METRIC_VIEW_TO_PIPELINE[metric_view])
        return True

    def _on_source_finished(self) -> None:
        """영상 하나가 끝났다 — 대조하고, 남은 시점이 있으면 이어서 돌린다."""
        self._view_transition_timer.stop()
        self._run_metric_check()
        if self._start_next_video_view():
            return
        self._show_metric_comparison()
        if self._session_reps:
            self._show_game_result()
        else:
            self.status_label.setText(
                "영상이 끝났지만 완료된 스쿼트를 찾지 못했습니다. "
                "전신이 화면에 들어오고 준비 자세에서 시작하는 영상인지 확인해 주세요."
            )

    def _advance_view(self) -> None:
        if self._video_path:
            # 영상 한 개에는 한 시점만 담겨 있다 — 다음 시점으로 넘어가지 않는다.
            self._on_source_finished()
            return
        if self._view_index + 1 >= len(VIEW_SEQUENCE):
            return
        self._view_index += 1
        self._start_view_worker(VIEW_SEQUENCE[self._view_index])

    def stop(self) -> None:
        self._countdown_timer.stop()
        self._ref_playback_timer.stop()
        self._rep_popup_timer.stop()
        self._view_transition_timer.stop()
        self.countdown_label.hide()
        self.rep_popup_label.hide()
        self.rep_counter_label.hide()
        self._phase_complete_hold_timer.stop()
        self._set_phase_stage(None)
        self.game_result_panel.hide()
        running_worker = self.worker if self.worker is not None and self.worker.isRunning() else None
        if running_worker is not None:
            running_worker.requestInterruption()
            if not running_worker.wait(5000) and self._session_recording is not None:
                # 아직 영상 파일을 쓰는 중이면 QThread 종료 후 임시 디렉터리를 정리한다.
                recording = self._session_recording
                running_worker.finished.connect(recording.close)
                self._session_recording = None
        self.worker = None
        if self._session_recording is not None:
            self._session_recording.close()
            self._session_recording = None

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
        ref_xy = self.ref_track.step(self._current_view)
        canvas = np.zeros((REF_PANEL_H, REF_PANEL_W, 3), dtype=np.uint8)
        draw_skeleton_panel(
            canvas, (0, 0), REF_PANEL_W, REF_PANEL_H, self._ref_tf(ref_xy),
            f"{VIEW_LABEL_KO[self._current_view]} 정상 레퍼런스 · 기준 속도 ({self.ref_track.phase_at()})", None,
            COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
        )
        self.ref_panel.set_bgr_frame(canvas)

    def _update_my_skeleton_panel(self, status: PipelineStatus) -> None:
        """보정된 표시용 3D를 그린다. 판정용 원본 정렬 좌표와는 구분한다.

        정규화 축과 대략적인 다리 스케일은 레퍼런스와 공유하지만, 측면 다리는
        카메라 평면의 2D 관절 위치로 보이는 형태를 보정한다.
        """
        if status.aligned_frame is None:
            return
        phase_text = (PHASE_LABEL_KR.get(status.phase, "-") if status.framing_ok
                      else "판정 보류")
        self.my_skeleton_panel.set_points(status.aligned_frame, phase_text)

    def _update_countdown(self, status: PipelineStatus) -> None:
        if self.worker is not None and self.worker.session_active:
            return  # 이미 시작됨, 더 이상 카운트다운 로직 필요 없음
        if self._video_path:
            # 사전 녹화 영상은 "자리를 잡을 시간"을 줄 대상이 아니다. 카운트다운을 두면
            # 화각 판정이 한 프레임만 흔들려도 취소-재시작을 반복해 영원히 시작되지 않는다.
            if self.worker is not None:
                self.worker.session_active = True
            self.countdown_label.hide()
            return
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

    def _collect_metric_frame(self, status: PipelineStatus) -> None:
        """정규화 전 좌표를 그대로 모아 둔다 — 영상이 끝나면 정답 CSV와 대조한다."""
        if not self._video_path:
            return
        self._metric_pred.append(
            status.common3d.copy() if status.common3d is not None
            else np.full((len(COMMON_JOINT_NAMES), 3), np.nan)
        )

    def _run_metric_check(self) -> None:
        """모아둔 추정 좌표를 AI Hub 3D 정답과 1:1 대조해 시점별 매칭율을 띄운다."""
        if not self._metric_pred:
            return
        candidates = ([self._label_archive_path] if self._label_archive_path else [])
        candidates += [str(c) for c in find_label_archives()]
        try:
            clip = parse_source_video(Path(self._video_path))
            archive = resolve_ground_truth(clip, candidates)
            if archive is None:
                return  # AI Hub 자료가 아니거나 정답이 없으면 그냥 재생만 한 것이다
            truth = archive.ground_truth_common3d(clip)
            report = compare_sequences(
                np.stack(self._metric_pred), truth,
                self._metric_view or clip.view or "front",
            )
        except MatchError as error:
            self._show_match_error(error)
            return
        self._metric_reports.append((clip, report))
        self.metric_label.setText(
            f"메트릭 검증 진행 중 · {clip.describe()}\n"
            f"{METRIC_VIEW_LABEL_KO.get(report.view, report.view)} "
            f"매칭율 {report.match_rate:.1f}% · 평균오차 {report.pa_mpjpe_mm:.0f}mm"
        )
        self.metric_label.show()

    def _show_metric_comparison(self) -> None:
        """세 시점을 다 돌고 나서 시점별 매칭율을 나란히 비교한다."""
        if not self._metric_reports:
            return
        clip = self._metric_reports[0][0]
        lines = [
            f"메트릭 검증 결과 · {clip.error_type}/{clip.level}/{clip.actor}/rep{clip.rep} · "
            f"3D 소스: {self._depth_source}",
            f"{'시점':<6}{'매칭율':>9}{'평균오차':>11}{'대조프레임':>12}   오차 큰 관절",
        ]
        best = max(self._metric_reports, key=lambda item: item[1].match_rate)
        for _clip, report in self._metric_reports:
            worst = sorted(report.per_joint_mm.items(), key=lambda kv: -kv[1])[:2]
            mark = " ◀ 최고" if report is best[1] else ""
            lines.append(
                f"{METRIC_VIEW_LABEL_KO.get(report.view, report.view):<6}"
                f"{report.match_rate:>8.1f}%"
                f"{report.pa_mpjpe_mm:>9.0f}mm"
                f"{report.detected_frames:>7}/{report.frames:<4}"
                f"   " + ", ".join(f"{k} {v:.0f}mm" for k, v in worst) + mark
            )
        rates = [r.match_rate for _c, r in self._metric_reports]
        errors = [r.pa_mpjpe_mm for _c, r in self._metric_reports]
        lines.append(
            f"평균 매칭율 {sum(rates) / len(rates):.1f}% · "
            f"평균 오차 {sum(errors) / len(errors):.0f}mm · "
            f"허용오차 {self._metric_reports[0][1].threshold_mm:.0f}mm 기준"
        )
        self.metric_label.setText("\n".join(lines))
        self.metric_label.show()

    def _on_status(self, status: PipelineStatus) -> None:
        self._collect_metric_frame(status)
        # 시점 전환 직전 워커에서 큐에 남은 신호가 다음 측정에 섞이지 않게 폐기한다.
        if self._transitioning_view or status.view_mode != self._current_view:
            return
        view_label = VIEW_LABEL_KO[self._current_view]
        self.camera_panel.set_bgr_frame(status.video_bgr)
        self._update_my_skeleton_panel(status)
        self._update_phase_stepper(status)
        self.fps_label.setText(f"{status.fps:4.1f} FPS")
        self.rep_label.setText(f"{view_label} REP {status.rep_count} / {REPS_PER_VIEW}")

        if status.pose_found and status.framing_ok:
            conf_flag = f" [인식불안 {status.n_frozen}/18]" if status.n_frozen >= 6 else ""
            self.status_label.setText(
                f"● {view_label} 자세 감지됨 (phase: {PHASE_LABEL_KR.get(status.phase, '-')}){conf_flag}"
            )
            self.status_label.setStyleSheet("color: #72df8d; font-weight: 600;")
        elif status.pose_found:
            self.status_label.setText(f"⚠ {view_label} 위치 조정 필요")
            self.status_label.setStyleSheet("color: #f2bd61; font-weight: 600;")
        else:
            self.status_label.setText(f"○ {view_label} 전신 자세를 찾는 중…")
            self.status_label.setStyleSheet("color: #f2bd61; font-weight: 600;")

        self._update_countdown(status)
        self._update_joint_panel(status)

        # 우측 레퍼런스 패널은 이제 여기서 갱신하지 않는다 — _advance_reference_panel()이
        # 독립된 REFERENCE_FPS 타이머로 갱신한다(사용자 상태와 무관하게 정상 배속 유지).

        # 위치/화각/요청 시점이 데이터셋 조건에 안 맞으면 DTW 판정 대신
        # 위치 안내부터 보여준다 — 잘못된 위치에서 나온 "확인 필요"는 의미가 없다.
        if not status.framing_ok:
            self._set_judge(status.framing_message, "positioning")
            self._bad_streak_key = None
            self._bad_streak_count = 0
        elif status.partial_distance and status.partial_distance.get("two_stage_pending"):
            self._set_judge("동작 분석 중 · 반복 완료 후 판정", "neutral")
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
            if r.decision_stage == "legacy" and r.score_vs_normal is not None and r.score_vs_normal >= PASS_SCORE_THRESHOLD:
                display_class, confidence_note = "정상", ""
            else:
                display_class = r.predicted_class
                confidence_note = "" if r.decision_stage != "legacy" or margin > JUDGE_MARGIN_THRESHOLD else " (근소한 차이 — 참고용)"
            # Keep the established trajectory/ML judgment independent from the
            # paper's bottom-pose rules.  The combined display_class below is
            # used only for the short popup and session-level safety outcome.
            sequence_class = display_class

            assessment = r.condition_assessment or {}
            paper_violations = assessment.get("paper_posture_violations") or []
            paper_messages = [str(item.get("message", "논문 자세 기준 위반"))
                              for item in paper_violations]
            # The paper's three criteria are evaluated only from the visible
            # side at the squat bottom.  A failed criterion must be visible in
            # the REP result even when the sequence gate happened to pass.
            if paper_messages and display_class == "정상":
                display_class = "논문 자세 기준 위반"
            if self._current_view == VIEW_FRONT:
                paper_result = "미평가 — 정면에서는 전후 깊이·발끝 대비 무릎 위치를 판별할 수 없음"
            elif paper_messages:
                paper_result = "위반 — " + " / ".join(paper_messages)
            else:
                paper_result = "통과 — 고관절 각도·허벅지 수평·무릎-발끝 정렬"

            if r.gate_distance is not None and r.gate_threshold is not None:
                score_text = f"DTW {r.gate_distance:.3f} / 기준 {r.gate_threshold:.3f}"
            if r.body_part_probabilities:
                likely_parts = [name for name, probability in r.body_part_probabilities.items() if probability >= 0.5]
                part_text = ", ".join(likely_parts) if likely_parts else "부위 미확정"
            else:
                part_text = "-"
            stage_text = {
                "dtw_pass": "1차 DTW 통과",
                "mt_stgcn": "2차 그래프 모델 진단",
                "diagnosis_unavailable": "2차 진단 모델 없음",
                "side_model_unavailable": "측면 판정 모델 없음",
            }.get(r.decision_stage, "기존 판정")

            # 판정 근거: DTW가 실제로 비교에 쓴 요소(코드명 -> 한국어) + 관절별 실측
            # 각도/합격범위/초과분. "왜 이 판정이 나왔는지 숫자로 보여달라"는 실사용
            # 피드백 대응 — judge_label/팝업의 클래스명만으론 검증할 방법이 없었다.
            feature_labels = [FEATURE_LABEL_KR.get(name, name) for name, _ in r.top_contributing_features]
            joint_lines = [f"{js.name} {joint_detail_text(js)}" for js in (self._latest_joint_scores or [])]
            joint_summary = "  |  ".join(joint_lines) if joint_lines else "-"

            self.result_label.setText(
                f"REP 종료\n기존 궤적·ML 판정: {sequence_class}{confidence_note} ({stage_text})  |  판정 지표: {score_text}  |  "
                f"주요 원인: {', '.join(feature_labels)}  |  모델 추정 부위: {part_text}\n"
                f"관절별 근거: {joint_summary}\n"
                f"논문 최저점 자세 기준: {paper_result}"
            )
            # 오류 REP에 대한 구체적인 원인 문구 — 벗어난 관절이 있으면 그 관절의 실측
            # 각도/합격범위/초과분을 그대로 쓰고(가장 구체적), 없으면 DTW 주요 feature명으로
            # 대체한다. 팝업뿐 아니라 최종 결과화면(_show_game_result)에도 재사용한다
            # (요청사항: "엉덩이하방오류가 자세가 정확히 어떻게 문제됐는지 완료화면에 띄워줘").
            failing_js = [js for js in (self._latest_joint_scores or []) if not js.within_angle_tolerance]
            if sequence_class == "정상":
                fail_reason = ""
            elif failing_js:
                # 결과화면에 REP마다 다 나열되면 길어지니 최대 2개 관절만 구체적으로 보여준다.
                fail_reason = ", ".join(f"{js.name} {joint_detail_text(js)}" for js in failing_js[:2])
                if len(failing_js) > 2:
                    fail_reason += f" 외 {len(failing_js) - 2}개"
            else:
                fail_reason = feature_labels[0] if feature_labels else ""

            condition_error = assessment.get("predicted_error")
            condition_support = assessment.get("error_support") or {}
            if condition_error:
                support_pct = 100.0 * float(condition_support.get(condition_error, 0.0))
                condition_note = f"조건 근거: {condition_error} {support_pct:.0f}%"
                if sequence_class != "정상" and not fail_reason:
                    fail_reason = condition_note
            else:
                condition_note = "조건 근거: 반복 확인 필요"
            self.result_label.setText(self.result_label.text() + f"\n기존 판정 보조 조건: {condition_note}")

            view_rep_number = 1 + sum(
                rep["view_mode"] == self._current_view for rep in self._session_reps
            )
            global_index = len(self._session_reps)
            self._session_reps.append({
                "index": global_index,
                "view_rep_index": view_rep_number - 1,
                "view_mode": self._current_view,
                "display_class": display_class,
                "sequence_class": sequence_class,
                "paper_posture_result": paper_result,
                "paper_posture_messages": paper_messages,
                # 최종 융합에는 DL/DTW 시퀀스 분류기의 원 판정을 넣고, 화면용
                # 정상 유사도 보정 결과(display_class)와 구분한다.
                "model_class": r.predicted_class,
                "score_text": score_text,
                "fail_reason": fail_reason,
                "condition_assessment": assessment,
            })
            total_reps = len(self._session_reps)
            self.rep_counter_label.setText(
                f"{view_label} {view_rep_number}/{REPS_PER_VIEW} · 전체 {total_reps}/{TARGET_REPS}"
            )

            if total_reps >= TARGET_REPS:
                # 세 시점 각 3회, 총 9회를 채웠다 — 매 REP마다 뜨던 작은 팝업 대신, 확인을 누를
                # 때까지 안 사라지는 최종 결과 화면으로 넘어간다(요청사항).
                self._show_game_result()
            elif view_rep_number >= REPS_PER_VIEW:
                self._transitioning_view = True
                if self.worker is not None:
                    self.worker.requestInterruption()
                next_view = VIEW_SEQUENCE[self._view_index + 1]
                self._show_rep_popup(
                    f"{view_label} 3회 완료\n사용자의 {VIEW_LABEL_KO[next_view]}이 카메라를 향하도록 옮겨 주세요.",
                    "good",
                )
                self._view_transition_timer.start()
            elif display_class == "정상":
                self._show_rep_popup(f"성공! 👍\n{score_text}", "good")
            else:
                short_reason = f"{failing_js[0].name} 등 {len(failing_js)}개 관절 벗어남" if failing_js else fail_reason
                self._show_rep_popup(f"REP {r.rep_index + 1} 완료\n{display_class}\n{short_reason}", "bad")

    def _show_game_result(self) -> None:
        """세 시점의 TARGET_REPS를 채운 뒤 뜨는 결과 화면. 확인 버튼을 누르기 전까지 남아있고
        (요청사항), 카메라/레퍼런스 재생은 멈춰서 화면이 계속 바뀌지 않게 한다."""
        if self.worker is not None and self.worker.isRunning():
            self.worker.requestInterruption()
            if not self.worker.wait(5000):
                self.recording_result_label.setText("영상 파일을 마무리하는 중입니다. 잠시 후 저장해 주세요.")
                finishing_worker = self.worker
                finishing_worker.finished.connect(self._recording_worker_finished)
        if self.worker is not None and not self.worker.isRunning():
            self.worker = None
        self._ref_playback_timer.stop()
        self._rep_popup_timer.stop()
        self.rep_popup_label.hide()

        session_decision = decide_session(self._session_reps)
        lines = []
        for rep in self._session_reps:
            view_label = VIEW_LABEL_KO[rep["view_mode"]]
            sequence_class = rep.get("sequence_class", rep["display_class"])
            paper_result = rep.get("paper_posture_result")
            if paper_result is None:
                # Backward-compatible rendering for a recording produced
                # before the result fields were split.
                paper_result = "미평가" if rep["view_mode"] == VIEW_FRONT else "기록 없음"
            line = (
                f"{view_label} REP {rep['view_rep_index'] + 1}\n"
                f"  기존 궤적·ML 판정: {sequence_class} ({rep['score_text']})\n"
                f"  논문 최저점 자세 기준: {paper_result}"
            )
            if rep["fail_reason"]:
                line += f"\n  기존 판정 근거: {rep['fail_reason']}"
            lines.append(line)
        self._session_result_text = format_session_decision(session_decision) + "\n\n" + "\n".join(lines)
        self.game_result_title.setText(f"🏁 {TARGET_REPS}회 완료!")
        self.game_result_body.setText(self._session_result_text)
        self._pending_error_replay = True
        self.save_recording_btn.setEnabled(
            self._session_recording is not None and (self.worker is None or not self.worker.isRunning())
        )
        self.replay_error_btn.setEnabled(False)
        self._position_game_result_panel()
        self.game_result_panel.show()
        self.game_result_panel.raise_()
        if self.worker is None or not self.worker.isRunning():
            # Let the worker's finally block close the current MP4/JSONL before
            # opening it.  This also makes replay available for the final view.
            QTimer.singleShot(0, self._recording_worker_finished)

    def _recording_worker_finished(self) -> None:
        if self.worker is not None and not self.worker.isRunning():
            self.worker = None
        if (self.game_result_panel.isVisible() and self._session_recording is not None
                and (self.worker is None or not self.worker.isRunning())):
            self.save_recording_btn.setEnabled(True)
            self.recording_result_label.setText("녹화 영상과 분석자료를 저장할 수 있습니다.")
            try:
                has_error_video = bool(build_replay_views(
                    self._session_recording.directory, self._session_reps,
                ))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                has_error_video = False
            self.replay_error_btn.setEnabled(has_error_video)
            if self._pending_error_replay and not self._error_replay_shown:
                self._pending_error_replay = False
                QTimer.singleShot(0, self._show_recorded_error_replay)

    def _show_recorded_error_replay(self) -> None:
        """Show all recorded preparation-to-finish views after final judgement."""
        if self._session_recording is None:
            QMessageBox.warning(self, "세션 영상 다시보기", "재생할 세션 녹화가 없습니다.")
            return
        if self.worker is not None and self.worker.isRunning():
            self.recording_result_label.setText("녹화를 마무리한 뒤 전체 영상을 표시합니다.")
            self._pending_error_replay = True
            return
        try:
            replay_views = build_replay_views(self._session_recording.directory, self._session_reps)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            QMessageBox.warning(self, "오류 동작 영상", f"오류 영상 준비에 실패했습니다: {error}")
            return
        if not replay_views:
            QMessageBox.information(self, "오류 동작 영상", "최종 판정에서 확정된 오류 동작이 없습니다.")
            return
        self._error_replay_shown = True
        self.replay_error_btn.setEnabled(True)
        PostSessionReplayDialog(replay_views, self).exec_()

    def _save_recording(self) -> None:
        if self._session_recording is None:
            QMessageBox.warning(self, "녹화 저장", "저장할 세션 녹화가 없습니다.")
            return
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "녹화 저장", "카메라 녹화가 종료될 때까지 기다려 주세요.")
            return
        selected = QFileDialog.getExistingDirectory(
            self, "스쿼트 녹화 저장 폴더 선택", str(Path.home())
        )
        if not selected:
            return
        try:
            saved = self._session_recording.export(selected, {
                "exercise": "에어스쿼트",
                "camera_index": self._last_camera_index,
                "final_result": self._session_result_text,
                "repetitions": self._session_reps,
            })
        except (OSError, RuntimeError, ValueError, TypeError) as error:
            QMessageBox.warning(self, "녹화 저장 실패", str(error))
            return
        missing = [VIEW_LABEL_KO[view] for view in VIEW_SEQUENCE
                   if not (saved / f"{view}.json").is_file()]
        suffix = f" · 녹화되지 않은 시점: {', '.join(missing)}" if missing else ""
        self.recording_result_label.setText(f"저장 완료: {saved}{suffix}")

    def _on_retry_clicked(self) -> None:
        if self._last_class_label is None:
            return
        # 영상 모드로 시작했으면 같은 영상으로 다시 돌린다 — 입력 소스와 3D 추정 방식은
        # 재시도할 때마다 바뀌면 안 되는 "이번 세션의 설정"이다.
        self.start(
            self._last_class_label, self._last_medoid_rank, self._last_camera_index,
            self._video_path, self._depth_source, self._label_archive_path,
        )

    def _on_confirm_clicked(self) -> None:
        self.game_result_panel.hide()
        self._on_back()

    def _on_worker_message(self, source_view: str, message: str) -> None:
        if source_view == self._current_view and not self._transitioning_view:
            self.status_label.setText(message)

    def _on_worker_error(self, source_view: str, message: str) -> None:
        if source_view != self._current_view or self._transitioning_view:
            return
        self._on_error(message)
        # 9회 완료 전 오류에서도 이미 녹화된 프레임을 버리지 않고 내보낼 수 있게 한다.
        self._ref_playback_timer.stop()
        self._countdown_timer.stop()
        self._session_result_text = f"측정 중 오류: {message}\n완료한 반복: {len(self._session_reps)}/{TARGET_REPS}"
        self.game_result_title.setText("측정 중 오류")
        self.game_result_body.setText(self._session_result_text)
        self.recording_result_label.setText("녹화 파일을 마무리하는 중입니다. 잠시 후 저장해 주세요.")
        if self.worker is not None and self.worker.isRunning():
            self.worker.finished.connect(self._recording_worker_finished)
        self.save_recording_btn.setEnabled(
            self._session_recording is not None and (self.worker is None or not self.worker.isRunning())
        )
        self._position_game_result_panel()
        self.game_result_panel.show()
        self.game_result_panel.raise_()

    def _on_error(self, message: str) -> None:
        self.status_label.setText(f"오류: {message}")
        self.status_label.setStyleSheet("color: #ff7b7b; font-weight: 600;")


__all__ = ["SelectionScreen", "CompareScreen"]
