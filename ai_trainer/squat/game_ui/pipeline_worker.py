"""Qt 워커 스레드: 카메라 + MediaPipe(ms.choe) + Common Skeleton + Lifting +
OnlineSquatSession(이 브랜치)을 한 스레드에서 순차 실행하고 신호로 내보낸다.

ms.choe의 `live_pose.worker.CameraPoseWorker`와 같은 패턴(카메라 오픈 로직,
FPS 스무딩, 에러 처리)을 따르되 파이프라인 뒷단(Common Skeleton 매핑 ->
2D->3D Lifting -> 정규화 -> Online Phase/DTW)을 추가한다. `FrameProcessor`를
그대로 쓰지 않고 detector.process()를 직접 호출하는 이유는 표시용 프레임과
이 브랜치의 DTW 파이프라인이 **같은 PoseObservation**(2중 추론 방지)을
공유해야 하기 때문이다.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from ai_trainer.core.s1_capture.camera_calibration import (
    CameraCalibration,
    CameraCalibrationError,
    FrameUndistorter,
    pixels_to_unit_rays,
    unmirror_pixels,
)
from ai_trainer.squat.camera_views import VIEW_FRONT, VIEWS
from ai_trainer.squat.dl_classifier import DLSquatClassifier
from ai_trainer.squat.mt_stgcn import SquatErrorDiagnoser
from ai_trainer.squat.two_stage_squat import NormalTemplateGate
from ai_trainer.squat.joint_feedback import JointScore, compute_joint_scores, visible_joint_scores
from ai_trainer.core.s2_pose.mediapipe_pose import MediaPipePoseDetector, PoseBackendError
from ai_trainer.core.ui.pose_overlay import draw_2d_pose
from ai_trainer.core.s1_capture.camera_worker import CameraConfig, _open_camera
from ai_trainer.squat.depth_source import (
    SOURCE_MEDIAPIPE,
    LiftingDepthEstimator,
    combine as combine_depth,
    lifting_to_common3d,
)
from ai_trainer.squat.lifting_model import TemporalLiftingNet
from ai_trainer.squat.online_dtw import OnlineSquatSession
from ai_trainer.squat.reference_db_io import load_reference_db
from ai_trainer.squat.scoring import PASS_SCORE_THRESHOLD, distance_to_score
from ai_trainer.squat.view_conditions import assess_view_rep, load_view_condition_config
from ai_trainer.core.s3_mapping.common_skeleton import COMMON_JOINT_NAMES as _COMMON_NAMES

from ai_trainer.squat.game_ui.framing_check import (
    check_framing,
    guide_box as compute_guide_box,
    upright_calibration_pose,
)
from ai_trainer.squat.game_ui.session_recording import ViewRecorder

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
LIFTING_CKPT = ROOT / "output" / "lifting_baseline" / "model_best.pt"
WEIGHTS_CFG_PATH = ROOT / "configs" / "dtw_feature_weights.json"
DB_DIR = ROOT / "output" / "reference_db"
OFFLINE_REPORT_PATH = ROOT / "output" / "dtw_eval" / "offline_eval_report.json"
DL_CLASSIFIER_CKPT = ROOT / "output" / "dl_classifier" / "model.pt"
DL_CLASSIFIER_NORM = ROOT / "output" / "dl_classifier" / "norm_stats.npz"
TWO_STAGE_DIR = ROOT / "output" / "two_stage_squat"
SIDE_GATE_PATH = TWO_STAGE_DIR / "side_view_gates.npz"
SIDE_MODEL_PATH = TWO_STAGE_DIR / "side_view_mt_stgcn.pt"
DEFAULT_CAMERA_CALIBRATION_PATH = ROOT / "configs" / "local_camera_calibration.json"
_NECK_IDX = _COMMON_NAMES.index("Neck")
_HIP_IDX = _COMMON_NAMES.index("Hip")
_LHIP_IDX = _COMMON_NAMES.index("LHip")
_RHIP_IDX = _COMMON_NAMES.index("RHip")
# 화면용 방향 정렬에 쓸 표본 수. 서 있는 초반 프레임의 골반축 중앙값을 쓰면
# 한두 프레임이 흔들려도 방향이 튀지 않는다.
_DISPLAY_YAW_SAMPLES = 15


@dataclass(frozen=True)
class PipelineStatus:
    video_bgr: np.ndarray
    fps: float
    pose_found: bool
    mean_confidence: float
    n_frozen: int
    framing_ok: bool
    framing_message: str
    phase: str | None
    pelvis_height: float | None  # 정규화(leg_length 단위) pelvis 높이 -> 레퍼런스 동기화에 사용
    rep_count: int
    partial_distance: dict | None  # {"phase":..., "distance_by_class": {...}}
    completed_rep: object | None  # ai_trainer.squat.online_dtw.RepResult
    live_score: float | None  # partial_distance["정상"]을 score_calib으로 환산한 실시간 0~100 유사도(%)
    joint_scores: list[JointScore] | None  # 관절별 위치/각도 오차 (joint_feedback.compute_joint_scores)
    aligned_frame: np.ndarray | None  # 판정 입력과 분리된 표시용 보정 3D (18,3)
    view_mode: str  # front / left / right
    # 정규화 전 Common Skeleton 3D (18,3). 메트릭 검증에서 AI Hub 정답과 대조할 때
    # 회전·스케일 정합 외의 변수를 넣지 않으려고 이 원본 좌표를 쓴다.
    common3d: np.ndarray | None = None


class SquatPipelineWorker(QThread):
    status_ready = pyqtSignal(object)  # PipelineStatus
    status_changed = pyqtSignal(str)
    fatal_error = pyqtSignal(str)
    source_finished = pyqtSignal()  # 사전 녹화 영상이 끝까지 재생됨 (카메라 입력에는 없음)

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        config: CameraConfig | None = None,
        view_mode: str = VIEW_FRONT,
        recording_dir: str | Path | None = None,
        debug_tap=None,
        depth_source: str = SOURCE_MEDIAPIPE,
        record_view: str = "",
        parent=None,
    ):
        super().__init__(parent)
        if view_mode not in VIEWS:
            raise ValueError(f"지원하지 않는 촬영 시점입니다: {view_mode}")
        self.model_path = Path(model_path).resolve()
        self.config = config or CameraConfig()
        self.view_mode = view_mode
        self.recording_dir = Path(recording_dir) if recording_dir is not None else None
        # 디버깅 모드에서만 주어지는 ai_trainer.debug.StageTap. None이면 공정 스냅샷을 만들지 않는다.
        self.debug_tap = debug_tap
        # 3D를 어디서 얻을지 (ai_trainer.squat.depth_source 참고). 정면 영상에서
        # 상체 기울기를 살리려면 lifting/hybrid가 필요하다.
        self.depth_source = depth_source
        # 녹화 파일 이름과 화면 표기에 쓰는 실제 시점 이름. AI Hub 사선(camera2)처럼
        # 판정 규칙이 없는 각도는 view_mode를 front로 돌리되 기록은 'oblique'로 남겨야
        # 정면 녹화를 덮어쓰지 않는다.
        self.record_view = record_view or view_mode
        # 3-2-1 카운트다운이 끝나기 전까지는 False로 두어 캘리브레이션/phase/DTW가 시작되지
        # 않게 한다. 메인(UI) 스레드에서 True로 바꿔주면 그 다음 프레임부터 세션이 시작된다.
        # 단순 bool 속성 읽기/쓰기라 CPython GIL 하에서 스레드 간 공유에 안전하다
        # (QThread.isInterruptionRequested()와 같은 패턴).
        self.session_active = False
        # framing_check는 매 프레임 독립적으로 계산되는데, MediaPipe 추정이 한 프레임만
        # 살짝 흔들려도(예: 발뒤꿈치 visibility가 잠깐 0.4 밑으로) 바로 "오류"로 튀면
        # 실제로는 잘 서 있는데도 판정이 계속 깜빡여 진행이 안 되는 문제가 있었다.
        # 그래서 상태를 몇 프레임 연속으로 같은 방향일 때만 실제로 전환한다(히스테리시스).
        self._framing_effective_ok = False
        self._framing_streak = 0

    FRAMING_DEBOUNCE_FRAMES = 5

    def run(self) -> None:
        capture = None
        detector = None
        view_recorder = None
        try:
            try:
                import cv2
            except ImportError as error:
                raise RuntimeError("OpenCV가 설치되지 않았습니다. pip install -r requirements.txt") from error

            self.status_changed.emit("모델을 불러오는 중…")
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

            lifting_model = TemporalLiftingNet(n_joints=18, hidden=128)
            lifting_model.load_state_dict(torch.load(LIFTING_CKPT, map_location=device))
            lifting_model.to(device).eval()

            weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
            db = load_reference_db(DB_DIR)
            score_calib = None
            if OFFLINE_REPORT_PATH.exists():
                # ground_truth tier 전용 calibration을 쓴다 (아래 3D bridge 설명 참고).
                # A CSV-only offline report does not include this optional
                # real-video calibration.  DTW judgement remains available.
                report = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))
                score_calib = report.get("score_calibration_ground_truth")
                if score_calib is None:
                    print(
                        "[warning] Real-video score calibration is unavailable; "
                        "continuing with DTW judgement without a percentage score."
                    )

            # Keep the earlier four-class model only for older installations.
            dl_classifier = None
            if not (TWO_STAGE_DIR / "normal_template.npz").exists() and DL_CLASSIFIER_CKPT.exists() and DL_CLASSIFIER_NORM.exists():
                dl_classifier = DLSquatClassifier.load(DL_CLASSIFIER_CKPT, DL_CLASSIFIER_NORM)
            elif not (TWO_STAGE_DIR / "normal_template.npz").exists():
                self.status_changed.emit(
                    "DL 분류기 체크포인트가 없어 DTW 판정으로 실행합니다."
                )

            normal_gate = None
            error_diagnoser = None
            if self.view_mode == VIEW_FRONT and (TWO_STAGE_DIR / "normal_template.npz").exists():
                normal_gate = NormalTemplateGate.load(TWO_STAGE_DIR / "normal_template.npz")
                if (TWO_STAGE_DIR / "mt_stgcn.pt").exists():
                    error_diagnoser = SquatErrorDiagnoser.load(TWO_STAGE_DIR / "mt_stgcn.pt")
                else:
                    self.status_changed.emit("오류 진단 모델이 없어 DTW 통과 여부만 판단합니다.")
            elif self.view_mode != VIEW_FRONT:
                if SIDE_GATE_PATH.exists():
                    normal_gate = NormalTemplateGate.load_side(SIDE_GATE_PATH, self.view_mode)
                    if SIDE_MODEL_PATH.exists():
                        error_diagnoser = SquatErrorDiagnoser.load(SIDE_MODEL_PATH)
                    else:
                        self.status_changed.emit("측면 오류 진단 모델이 없어 불확실 판정으로 처리합니다.")
                else:
                    self.status_changed.emit("측면 전용 정상 기준이 없어 불확실 판정으로 처리합니다.")

            # 실시간 3D 소스: 자체 학습한 lifting 모델(model) 대신 MediaPipe 자체
            # world_landmarks를 쓴다 — 실제 아이폰 촬영 영상 검증에서, 학습 분포 밖
            # 체형/팔자세/화각을 만나면 lifting 모델이 스쿼트 깊이를 심하게 과소평가하는
            # 것이 확인됐다(2026-08-28. pose_bridge.CommonSkeleton3DBridge 참고). 그래서
            # 비교 대상도 그 모델이 재현된 "operational" tier가 아니라, 8카메라 삼각측량
            # 실측 3D인 "ground_truth" tier로 맞춘다 — 둘 다 이제 "실제 3D" 도메인이라
            # lifting 모델을 거치지 않고도 서로 비교 가능하다. lifting_model은 AI Hub
            # CSV 기반 오프라인 평가/합성 테스트 경로에서만 여전히 쓰인다(그쪽은 원본
            # 영상이 없어 2D CSV -> 3D 복원이 유일한 방법).
            session = OnlineSquatSession(
                model=lifting_model, device=device, db_operational=db["ground_truth"],
                weights_cfg=weights_cfg, score_calib=score_calib, dl_classifier=dl_classifier,
                normal_gate=normal_gate, error_diagnoser=error_diagnoser,
                view_mode=self.view_mode,
            )

            from ai_trainer.core.s3_mapping.pose_bridge import CommonSkeleton3DBridge, CommonSkeletonBridge

            bridge = CommonSkeletonBridge(min_visibility=self.config.confidence)
            bridge3d = CommonSkeleton3DBridge(min_visibility=self.config.confidence)
            lifting_estimator = None
            if self.depth_source != SOURCE_MEDIAPIPE:
                from ai_trainer.squat.online_dtw import WINDOW_T

                lifting_estimator = LiftingDepthEstimator(
                    lifting_model, device, WINDOW_T, calib_frames=session.calib_frames
                )
                self.status_changed.emit(f"3D 소스: {self.depth_source}")
            common2d_history: list[np.ndarray] = []
            try:
                view_condition_config = load_view_condition_config()
            except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError):
                view_condition_config = None

            calibration_path = (
                Path(self.config.calibration_path)
                if self.config.calibration_path is not None
                else DEFAULT_CAMERA_CALIBRATION_PATH
            )
            undistorter = None
            if calibration_path.exists():
                try:
                    calibration = CameraCalibration.load(calibration_path)
                    calibration.require_camera_index(self.config.camera_index)
                    undistorter = FrameUndistorter(calibration, alpha=self.config.calibration_alpha)
                    self.status_changed.emit(
                        f"카메라 왜곡 보정 적용 (RMS {calibration.rms_reprojection_error:.2f}px)"
                    )
                except CameraCalibrationError as error:
                    self.status_changed.emit(f"카메라 보정 미적용: {error}")

            detector = MediaPipePoseDetector(
                self.model_path,
                min_detection_confidence=self.config.confidence,
                min_presence_confidence=self.config.confidence,
                min_tracking_confidence=self.config.confidence,
            )

            is_video = self.config.is_video
            self.status_changed.emit("영상을 여는 중…" if is_video else "카메라를 여는 중…")
            capture = _open_camera(cv2, self.config)
            # 영상은 읽는 만큼 즉시 나오므로, 카메라와 같은 속도로 흐르도록 원본 fps에
            # 맞춰 프레임 간격을 맞춘다(안 그러면 몇 배속으로 지나가 phase 판정이 깨진다).
            video_frame_interval = 0.0
            if is_video:
                source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
                video_frame_interval = 1.0 / source_fps if source_fps > 1.0 else 1.0 / 30.0
            self.status_changed.emit("실행 중")
            if self.recording_dir is not None:
                view_recorder = ViewRecorder(
                    self.recording_dir, self.record_view, cv2,
                    camera_index=self.config.camera_index,
                    calibration_applied=undistorter is not None,
                )

            previous_time = time.perf_counter()
            smoothed_fps = 0.0
            consecutive_failures = 0
            tap = self.debug_tap
            # 촬영 각도가 달라도 화면에서는 늘 같은 방향으로 서 있게, 수직축 회전(yaw)만
            # 한 번 구해 고정한다. 수직축 회전은 자세를 왜곡하지 않고 몸을 돌리기만 한다
            # (R_body 전체 정렬은 측면에서 축이 틀어져 스켈레톤이 꺾여 보였다).
            display_yaw = None
            yaw_samples: list[float] = []
            timing = {"read": 0.0, "mediapipe": 0.0, "rest": 0.0}
            loop_start_prev = None
            t_read = t_mp = 0.0

            while not self.isInterruptionRequested():
                t_loop = time.perf_counter()
                if tap is not None and loop_start_prev is not None:
                    total = t_loop - loop_start_prev
                    rest = total - t_read - t_mp
                    for key, value in (("read", t_read), ("mediapipe", t_mp), ("rest", rest)):
                        ms = 1000 * value
                        timing[key] = ms if timing[key] == 0.0 else timing[key] * 0.9 + ms * 0.1
                loop_start_prev = t_loop
                success, frame_bgr = capture.read()
                t_read = time.perf_counter() - t_loop
                if (not success or frame_bgr is None) and is_video:
                    self.status_changed.emit("영상 재생 완료")
                    self.source_finished.emit()
                    break
                if not success or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        raise RuntimeError("카메라 프레임을 연속으로 읽지 못했습니다.")
                    self.msleep(10)
                    continue
                consecutive_failures = 0

                # Camera intrinsics describe the raw sensor frame, so undistort it
                # before applying a selfie mirror or sending it to MediaPipe.
                if undistorter is not None:
                    try:
                        frame_bgr = undistorter.undistort(frame_bgr)
                    except CameraCalibrationError as error:
                        self.status_changed.emit(f"카메라 보정 중지: {error}")
                        undistorter = None
                display_bgr = np.ascontiguousarray(frame_bgr[:, ::-1] if self.config.mirror else frame_bgr)
                t_mp_start = time.perf_counter()
                observation = detector.process(np.ascontiguousarray(display_bgr[:, :, ::-1]))

                now = time.perf_counter()
                t_mp = now - t_mp_start
                instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                smoothed_fps = instantaneous_fps if smoothed_fps == 0.0 else smoothed_fps * 0.90 + instantaneous_fps * 0.10

                phase = None
                partial = None
                completed = None
                pelvis_height = None
                mean_conf = 0.0
                n_frozen = 0
                live_score = None
                joint_scores = None
                aligned_frame = None
                analysis_aligned_frame = None
                common2d = None
                common3d = None
                frozen3d = None
                analysis_status = None
                camera_rays = None
                effective_camera_matrix = None
                pose_found = observation is not None
                framing_ok = False
                framing_message = "카메라 앞에 서주세요"
                h, w = display_bgr.shape[:2]
                gbox = compute_guide_box(w, h)

                if pose_found:
                    video_bgr = draw_2d_pose(display_bgr, observation.image_landmarks)
                    # session_active(3-2-1 카운트다운 통과) 시점엔 준비 자세에서 거리가
                    # 이미 확인된 상태이므로, 이후엔 거리 재체크(relax_distance)를 건너뛴다
                    # — 안 그러면 스쿼트 최대 하강 지점에서 몸통이 접히며 화면상 세로
                    # 길이가 줄어드는 걸 "카메라에서 멀어졌다"로 오판정한다(실사용 재현 확인).
                    framing = check_framing(
                        observation.image_landmarks,
                        w,
                        h,
                        relax_distance=self.session_active,
                        view=self.view_mode,
                    )
                    framing_message = framing.message

                    # 히스테리시스: 판정이 바뀌는 방향으로 연속 N프레임 나와야 실제로 전환.
                    # 한 프레임만 흔들려도 바로 오류로 튀는 것을 막아준다.
                    if framing.ok == self._framing_effective_ok:
                        self._framing_streak = 0
                    else:
                        self._framing_streak += 1
                        if self._framing_streak >= self.FRAMING_DEBOUNCE_FRAMES:
                            self._framing_effective_ok = framing.ok
                            self._framing_streak = 0
                    framing_ok = self._framing_effective_ok

                    box_color = (90, 220, 90) if framing_ok else (60, 60, 240)
                    cv2.rectangle(video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), box_color, 2)
                    if framing.body_box is not None:
                        cv2.rectangle(video_bgr, (framing.body_box[0], framing.body_box[1]), (framing.body_box[2], framing.body_box[3]), (0, 200, 255), 1)
                    if not framing_ok:
                        cv2.putText(video_bgr, framing.message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 240), 2, cv2.LINE_AA)
                        if framing.low_confidence_joints:
                            # 어떤 관절이 구체적으로 안 잡히는지 진단용으로 표시
                            joints_str = ", ".join(f"{name}({v:.2f})" for name, v in framing.low_confidence_joints)
                            cv2.putText(video_bgr, f"인식 약함: {joints_str}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1, cv2.LINE_AA)

                    # 추적은 판정 게이트와 분리한다. 순간적인 화각 실패로 브릿지까지
                    # 멈추면 마지막 자세가 화면에 남고 복귀할 때 관절이 튄다.
                    common2d, frozen_mask, mean_conf = bridge.update(observation.image_landmarks, w, h)
                    common3d, frozen3d, _mean_conf3d = bridge3d.update(observation.world_landmarks)
                    if lifting_estimator is not None:
                        lifted = lifting_estimator.push(common2d)
                        if lifted is None:
                            # 창이 찰 때까지는 판정에 넘기지 않는다 — 섞인 좌표계가
                            # 캘리브레이션에 들어가면 R_body가 틀어진다.
                            continue
                        _center, lifted_raw = lifted
                        common3d = combine_depth(
                            self.depth_source, common3d,
                            lifting_to_common3d(lifted_raw, common3d),
                        )
                    n_frozen = int(frozen_mask.sum())

                    if tap is not None:
                        thumb_w = 320
                        thumb = cv2.resize(display_bgr, (thumb_w, max(1, int(thumb_w * h / w))))
                        tap.put("s1_capture", draw_2d_pose(thumb, observation.image_landmarks),
                                note=f"평균 신뢰도 {mean_conf:.2f}")
                        tap.put("s2_pose_3d", observation.world_landmarks[:, :3])
                        tap.put("s3_common_2d", common2d, note=f"freeze {n_frozen}", frame_size=(w, h))
                        tap.put("s3_common_3d", common3d, note=f"freeze {int(frozen3d.sum())}")

                    # Keep calibrated ray features in recordings for a future
                    # paired webcam 2D/3D lifting evaluation.  They are not fed
                    # into DTW/ML yet: the validated live source remains
                    # MediaPipe world landmarks.  MediaPipe saw the selfie-
                    # mirrored image, whereas K belongs to the physical camera,
                    # so undo mirroring before K^-1 [u,v,1].
                    if view_recorder is not None and undistorter is not None:
                        try:
                            effective_camera_matrix = undistorter.camera_matrix_for_undistorted_size((w, h))
                            ray_pixels = (
                                unmirror_pixels(common2d, w) if self.config.mirror else common2d
                            )
                            camera_rays = pixels_to_unit_rays(
                                ray_pixels, effective_camera_matrix
                            ).astype(np.float32)
                        except CameraCalibrationError as error:
                            # Do not interrupt a workout or silently use a raw
                            # K.  A later frame can retry after a resolution or
                            # capture-mode transition.
                            self.status_changed.emit(f"카메라 ray 기록 미적용: {error}")

                    if framing_ok and self.session_active:
                        # 화각/거리/정면 여부가 학습 데이터(AI Hub camera1)와 맞고, 3-2-1 카운트다운이
                        # 끝나 세션이 명시적으로 시작된 뒤에만 phase/DTW 파이프라인을 진행한다.
                        # 그렇지 않으면 잘못된 프레임이나 아직 자리를 잡는 중인 프레임이 session의
                        # 캘리브레이션/phase 상태기계에 섞여 들어가지 않도록 건너뛴다.
                        # common2d는 화면에 그릴 위치(말풍선/관절 색상 오버레이)용으로만 쓰고,
                        # 실제 phase/DTW 판정은 common3d(MediaPipe 자체 3D)로 한다.
                        # OnlineSquatSession의 frame index와 동일한 순서로 보관한다.
                        # REP 종료 시 동일 frame_range를 잘라 시점별 2D 조건을 평가한다.
                        common2d_history.append(common2d.copy())
                        calibration = upright_calibration_pose(
                            observation.image_landmarks, self.view_mode,
                        ) if session.R_body is None else None
                        status = session.push_frame_3d(
                            common3d,
                            calibration_ready=calibration.ok if calibration is not None else True,
                        )
                        analysis_status = status
                        if status is not None and status.get("status") != "ok" and calibration is not None:
                            samples = int(status.get("calibration_samples", 0))
                            framing_message = (
                                f"{calibration.message} ({samples}/{session.calib_frames})"
                            )
                        if status is not None and status.get("status") == "ok":
                            phase = status["phase"]
                            partial = status["partial_distance"]
                            pelvis_height = status["pelvis_height"]
                            analysis_aligned_frame = status["aligned_frame"]
                            if status["event"] == "rep_end":
                                completed = status["completed_rep"]
                                completed.view_mode = self.view_mode
                                if view_condition_config is not None:
                                    start, end = completed.frame_range
                                    if 0 <= start <= end < len(common2d_history):
                                        rep_2d = np.stack(common2d_history[start : end + 1])
                                        completed.condition_assessment = assess_view_rep(
                                            rep_2d,
                                            completed.phase_boundaries,
                                            self.view_mode,
                                            view_condition_config,
                                        ).to_dict()

                            if partial is not None and normal_gate is None and self.view_mode == VIEW_FRONT:
                                dvals = partial["distance_by_class"]
                                if score_calib is not None and "정상" in dvals:
                                    # REP 종료를 기다리지 않고, 진행 중인 phase의 "정상" 대비
                                    # distance를 그때그때 0~100 유사도로 환산해 실시간으로 보여준다.
                                    live_score = distance_to_score(dvals["정상"], score_calib)
                                best_class = min(dvals, key=dvals.get)
                                # "정상" 레퍼런스와의 절대 거리가 충분히 가까우면(유사도 %가
                                # PASS_SCORE_THRESHOLD 이상), 다른 오류 클래스가 근소하게
                                # 더 가깝다는 이유만으로 오류 말풍선을 띄우지 않는다 —
                                # judge_label(screens.py)과 동일한 기준으로 통일.
                                passes = live_score is not None and live_score >= PASS_SCORE_THRESHOLD
                                if best_class != "정상" and not passes:
                                    # 어떤 오류유형에 가장 가까운지에 따라 관련 관절 옆에 말풍선 설명을 붙인다
                                    # (예: 고관절오류 -> 고관절/상체 근처, claude.md 9장 오류유형별 feature 참고).
                                    pass  # final error overlay is rendered after the whole session

                            # 관절별 위치/각도 오차 (DTW와 별개 — "지금 이 순간" 프레임 레벨 피드백).
                            # session이 subsequence DTW로 찾아준 "지금 프레임에 대응하는 정상
                            # reference 프레임"(joint_feedback)을 받아 오차만 계산한다.
                            jf = status["joint_feedback"]
                            if jf is not None:
                                joint_scores = visible_joint_scores(
                                    compute_joint_scores(jf["user_frame"], jf["ref_frame"], phase=jf["phase"]),
                                    self.view_mode,
                                )
                else:
                    video_bgr = display_bgr.copy()
                    cv2.rectangle(video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (60, 60, 240), 2)

                if common3d is not None:
                    # 화면용 좌표는 회전시키지 않는다. R_body(몸 방향 정렬)는 측면·사선
                    # 촬영에서 좌우 hip 깊이가 불안정해 축이 틀어지고, 그러면 스켈레톤이
                    # 꺾이거나 좌우가 뒤집힌 것처럼 보인다(실측 확인). 보는 각도는
                    # OrbitSkeletonPanel에서 사용자가 직접 돌리면 되므로 여기서는
                    # 다리 길이로 크기만 맞추고 y축만 위로 뒤집어 넘긴다.
                    # scale3d는 캘리브레이션이 끝나야 생기는 속성이라 그 전에는 없다.
                    scale = getattr(session, "scale3d", None) or float(
                        np.linalg.norm(common3d[_NECK_IDX] - common3d[_HIP_IDX])
                    ) or 1.0
                    aligned_frame = common3d / scale
                    aligned_frame = aligned_frame * np.array([1.0, -1.0, 1.0])

                    if display_yaw is None:
                        hip_axis = aligned_frame[_LHIP_IDX] - aligned_frame[_RHIP_IDX]
                        if np.isfinite(hip_axis).all() and np.linalg.norm(hip_axis) > 1e-6:
                            yaw_samples.append(float(np.arctan2(hip_axis[2], hip_axis[0])))
                        if len(yaw_samples) >= _DISPLAY_YAW_SAMPLES:
                            display_yaw = float(np.median(yaw_samples))
                    if display_yaw is not None:
                        # 골반선이 화면 가로축(+x)을 향하도록 수직축 둘레로 돌린다.
                        cos_y, sin_y = np.cos(display_yaw), np.sin(display_yaw)
                        x = aligned_frame[:, 0] * cos_y + aligned_frame[:, 2] * sin_y
                        z = -aligned_frame[:, 0] * sin_y + aligned_frame[:, 2] * cos_y
                        aligned_frame = np.stack([x, aligned_frame[:, 1], z], axis=-1)

                if tap is not None:
                    if not pose_found:
                        for stage_id in ("s1_capture", "s2_pose_3d", "s3_common_2d", "s3_common_3d"):
                            tap.put(stage_id, None, note="사람 미검출")
                    if analysis_aligned_frame is None:
                        if not self.session_active:
                            wait_note = "세션 시작 전 (카운트다운 대기)"
                        elif not framing_ok:
                            wait_note = "화각 조건 미충족"
                        elif session.R_body is None:
                            wait_note = "준비 자세 캘리브레이션 중"
                        else:
                            wait_note = "판정 프레임 없음"
                        tap.put("s4_normalized", None, note=wait_note)
                    else:
                        tap.put("s4_normalized", analysis_aligned_frame)
                    tap.meta(
                        fps=smoothed_fps,
                        timing=dict(timing),
                        view=self.view_mode,
                        phase=phase,
                        status=None if framing_ok else framing_message,
                    )
                    tap.flush()

                # Keep preparation/calibration frames for the final full-view replay.
                if view_recorder is not None:
                    rep_event = None
                    if completed is not None:
                        rep_event = {
                            "rep_index": completed.rep_index,
                            "frame_range": completed.frame_range,
                            "predicted_class": completed.predicted_class,
                            "decision_stage": completed.decision_stage,
                            "gate_distance": completed.gate_distance,
                            "gate_threshold": completed.gate_threshold,
                            "raw_distance_by_class": completed.raw_distance_by_class,
                            "body_part_probabilities": completed.body_part_probabilities,
                            "condition_assessment": completed.condition_assessment,
                        }
                    diagnostics = {
                        "processing_fps": smoothed_fps,
                        "calibration_applied": undistorter is not None,
                        "camera_ray_feature_type": "camera_ray_v1" if camera_rays is not None else None,
                        "camera_ray_input": (
                            "undistorted physical/non-mirrored pixels" if camera_rays is not None else None
                        ),
                        "effective_camera_matrix": effective_camera_matrix,
                        "camera_rays": camera_rays,
                        "pose_found": pose_found,
                        "framing_ok": framing_ok,
                        "framing_message": framing_message,
                        "phase": phase,
                        "analysis_frame": analysis_status.get("session_frame") if analysis_status else None,
                        "analysis_event": analysis_status.get("event") if analysis_status else None,
                        "completed_rep": rep_event,
                        "image_landmarks": observation.image_landmarks if observation is not None else None,
                        "world_landmarks": observation.world_landmarks if observation is not None else None,
                        "common_2d": common2d,
                        "common_3d": common3d,
                        "frozen_3d": frozen3d,
                        "aligned_3d": analysis_aligned_frame,
                        "display_aligned_3d": aligned_frame,
                    }
                    try:
                        view_recorder.record(display_bgr, smoothed_fps, diagnostics)
                    except (OSError, RuntimeError, ValueError, TypeError) as error:
                        view_recorder.close()
                        view_recorder = None
                        self.status_changed.emit(f"녹화 중지: {error}")

                self.status_ready.emit(
                    PipelineStatus(
                        video_bgr=video_bgr,
                        fps=smoothed_fps,
                        pose_found=pose_found,
                        mean_confidence=mean_conf,
                        n_frozen=n_frozen,
                        framing_ok=framing_ok,
                        framing_message=framing_message,
                        phase=phase,
                        pelvis_height=pelvis_height,
                        rep_count=len(session.completed_reps),
                        partial_distance={"two_stage_pending": True} if partial is not None and normal_gate is not None else partial,
                        completed_rep=completed,
                        live_score=live_score,
                        joint_scores=joint_scores,
                        aligned_frame=aligned_frame,
                        view_mode=self.view_mode,
                        common3d=common3d,
                    )
                )

                if video_frame_interval > 0.0:
                    # 처리에 쓴 시간을 빼고 남은 만큼만 쉬어 원본 재생 속도를 맞춘다.
                    lag = video_frame_interval - (time.perf_counter() - t_loop)
                    if lag > 0:
                        self.msleep(int(lag * 1000))
        except (PoseBackendError, RuntimeError, ValueError, OSError) as error:
            self.fatal_error.emit(str(error))
        except Exception as error:  # noqa: BLE001
            self.fatal_error.emit(f"파이프라인 처리 중 예기치 않은 오류: {error}")
        finally:
            if view_recorder is not None:
                view_recorder.close()
            if capture is not None:
                capture.release()
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass


__all__ = ["SquatPipelineWorker", "PipelineStatus"]
