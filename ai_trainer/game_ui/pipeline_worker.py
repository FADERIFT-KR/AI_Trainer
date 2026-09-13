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
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector, PoseBackendError
from ai_trainer.live_pose.render import draw_2d_pose, render_3d_pose
from ai_trainer.live_pose.worker import CameraConfig, _open_camera
from ai_trainer.common_skeleton import COMMON_BONE_INDEX_PAIRS, COMMON_JOINT_NAMES
from ai_trainer.lifting_model import TemporalLiftingNet
from ai_trainer.online_dtw import OnlineSquatSession
from ai_trainer.reference_db_io import load_reference_db_by_level

from .framing_check import check_framing, check_pose_visibility, guide_box as compute_guide_box
from .error_explain import annotate_error
from .pose_bridge import world_to_common_skeleton
from ai_trainer.reference_levels import NORMAL_CLASS, REFERENCE_CLASSES

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
LIFTING_CKPT = ROOT / "output" / "lifting_baseline" / "model_best.pt"
WEIGHTS_CFG_PATH = ROOT / "configs" / "dtw_feature_weights.json"
DB_DIR = ROOT / "output" / "reference_db"
OFFLINE_REPORT_PATH = ROOT / "output" / "dtw_eval" / "offline_eval_report.json"
ERROR_VIDEO_LABELS = {
    REFERENCE_CLASSES[1]: "ERROR: HEEL POSITION",
    REFERENCE_CLASSES[2]: "ERROR: SQUAT DEPTH",
    REFERENCE_CLASSES[3]: "ERROR: HIP FLEXION",
}


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """세 관절 b에서 이루는 3D 각도(도)."""
    v1, v2 = a - b, c - b
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom < 1e-8:
        return float("nan")
    cosine = float(np.dot(v1, v2) / denom)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


@dataclass(frozen=True)
class PipelineStatus:
    video_bgr: np.ndarray
    skeleton_bgr: np.ndarray
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
    completed_rep: object | None  # ai_trainer.online_dtw.RepResult
    form_warnings: tuple[str, ...] = ()
    heel_lift_delta: tuple[float, float] | None = None
    ground_ready: bool = False
    completion_velocity_ok: bool = False
    completion_height_ok: bool = False
    skeleton_world: np.ndarray | None = None


class SquatPipelineWorker(QThread):
    status_ready = pyqtSignal(object)  # PipelineStatus
    status_changed = pyqtSignal(str)
    fatal_error = pyqtSignal(str)

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, config: CameraConfig | None = None, parent=None):
        super().__init__(parent)
        self.model_path = Path(model_path).resolve()
        self.config = config or CameraConfig()
        # 카운트다운 중에는 calibration_active만 켜서 준비 기준을 측정하고,
        # 카운트다운이 끝난 뒤 session_active를 켜서 phase/DTW를 시작한다.
        # 단순 bool 속성 읽기/쓰기라 CPython GIL 하에서 스레드 간 공유에 안전하다
        # (QThread.isInterruptionRequested()와 같은 패턴).
        self.session_active = False
        self.calibration_active = False
        self.calibration_ready = False
        self._calibration_request_id = 0
        # framing_check는 매 프레임 독립적으로 계산되는데, MediaPipe 추정이 한 프레임만
        # 살짝 흔들려도(예: 발뒤꿈치 visibility가 잠깐 0.4 밑으로) 바로 "오류"로 튀면
        # 실제로는 잘 서 있는데도 판정이 계속 깜빡여 진행이 안 되는 문제가 있었다.
        # 그래서 상태를 몇 프레임 연속으로 같은 방향일 때만 실제로 전환한다(히스테리시스).
        self._framing_effective_ok = False
        self._framing_streak = 0
        self._distance_locked = False
        self._initial_body_height_ratio: float | None = None
        self._error_feedback_class: str | None = None
        self._error_feedback_remaining = 0
        self._error_feedback_common2d: np.ndarray | None = None
        self._previous_world_skeleton: np.ndarray | None = None

    FRAMING_DEBOUNCE_FRAMES = 5

    def begin_countdown_calibration(self) -> None:
        """새 카운트다운에서 모든 준비 자세 기준값을 다시 측정한다."""
        self.session_active = False
        self.calibration_ready = False
        self.calibration_active = True
        self._calibration_request_id += 1

    def cancel_countdown_calibration(self) -> None:
        self.calibration_active = False
        self.calibration_ready = False

    def activate_session(self) -> None:
        """완료된 카운트다운 측정값을 고정하고 동작 판정을 시작한다."""
        self.calibration_active = False
        self.session_active = True

    def run(self) -> None:
        capture = None
        detector = None
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
            # 초급·중급·고급을 분리해 로드한다. OnlineSquatSession은 세 난이도의
            # 정상 레퍼런스 중 하나라도 충분히 유사하면 정상으로 판정한다.
            db = load_reference_db_by_level(DB_DIR)
            score_calib = None
            algorithm = "dtw"
            if OFFLINE_REPORT_PATH.exists():
                report = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))
                score_calib = report["score_calibration"]
                algorithm = report.get("algorithm", "dtw")

            session = OnlineSquatSession(
                model=lifting_model, device=device, db_operational=db["operational"],
                weights_cfg=weights_cfg, score_calib=score_calib,
                algorithm=algorithm,
            )

            from .pose_bridge import CommonSkeletonBridge

            bridge = CommonSkeletonBridge(min_visibility=self.config.confidence)

            detector = MediaPipePoseDetector(
                self.model_path,
                min_detection_confidence=self.config.confidence,
                min_presence_confidence=self.config.confidence,
                min_tracking_confidence=self.config.confidence,
            )

            self.status_changed.emit("카메라를 여는 중…")
            capture = _open_camera(cv2, self.config)
            self.status_changed.emit("실행 중")

            previous_time = time.perf_counter()
            smoothed_fps = 0.0
            consecutive_failures = 0
            handled_calibration_request_id = 0

            while not self.isInterruptionRequested():
                if handled_calibration_request_id != self._calibration_request_id:
                    session.reset_preparation_calibration()
                    bridge = CommonSkeletonBridge(min_visibility=self.config.confidence)
                    self._initial_body_height_ratio = None
                    self._distance_locked = False
                    self.calibration_ready = False
                    handled_calibration_request_id = self._calibration_request_id

                success, frame_bgr = capture.read()
                if not success or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        raise RuntimeError("카메라 프레임을 연속으로 읽지 못했습니다.")
                    self.msleep(10)
                    continue
                consecutive_failures = 0

                display_bgr = np.ascontiguousarray(frame_bgr[:, ::-1] if self.config.mirror else frame_bgr)
                observation = detector.process(np.ascontiguousarray(display_bgr[:, :, ::-1]))

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                smoothed_fps = instantaneous_fps if smoothed_fps == 0.0 else smoothed_fps * 0.90 + instantaneous_fps * 0.10

                phase = None
                partial = None
                completed = None
                pelvis_height = None
                completion_velocity_ok = False
                completion_height_ok = False
                skeleton_world = None
                joint_speed = None
                mean_conf = 0.0
                n_frozen = 0
                pose_found = observation is not None
                framing_ok = False
                framing_message = "카메라 앞에 서주세요"
                form_warnings: tuple[str, ...] = ()
                heel_lift_delta: tuple[float, float] | None = None
                ground_ready = False
                h, w = display_bgr.shape[:2]
                gbox = compute_guide_box(w, h)

                if pose_found:
                    common_world = world_to_common_skeleton(observation.world_landmarks)
                    skeleton_world = common_world
                    if self._previous_world_skeleton is not None:
                        joint_speed = float(np.nanmean(np.linalg.norm(common_world - self._previous_world_skeleton, axis=1)))
                    self._previous_world_skeleton = common_world.copy()
                    video_bgr = draw_2d_pose(display_bgr, observation.image_landmarks)
                    skeleton_bgr = render_3d_pose(
                        common_world,
                        width=640,
                        height=480,
                        connections=COMMON_BONE_INDEX_PAIRS,
                    )
                    # 레퍼런스와 DTW에서 사용하는 동일한 관절 각도를 카메라 화면에 표시한다.
                    # 좌/우 무릎: 골반-무릎-발목, 좌/우 고관절: 목-골반-무릎.
                    names = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
                    knee_l = _joint_angle_deg(common_world[names["LHip"]], common_world[names["LKnee"]], common_world[names["LAnkle"]])
                    knee_r = _joint_angle_deg(common_world[names["RHip"]], common_world[names["RKnee"]], common_world[names["RAnkle"]])
                    hip_l = _joint_angle_deg(common_world[names["Neck"]], common_world[names["LHip"]], common_world[names["LKnee"]])
                    hip_r = _joint_angle_deg(common_world[names["Neck"]], common_world[names["RHip"]], common_world[names["RKnee"]])
                    ankle_l = _joint_angle_deg(common_world[names["LKnee"]], common_world[names["LAnkle"]], common_world[names["LBigToe"]])
                    ankle_r = _joint_angle_deg(common_world[names["RKnee"]], common_world[names["RAnkle"]], common_world[names["RBigToe"]])
                    angle_text = (
                        f"Knee L/R: {knee_l:5.1f}/{knee_r:5.1f} deg  "
                        f"Hip L/R: {hip_l:5.1f}/{hip_r:5.1f} deg"
                    )
                    cv2.putText(video_bgr, angle_text, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (255, 220, 80), 1, cv2.LINE_AA)
                    ankle_text = f"Ankle L/R: {ankle_l:5.1f}/{ankle_r:5.1f} deg"
                    cv2.putText(video_bgr, ankle_text, (12, h - 42), cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (255, 220, 80), 1, cv2.LINE_AA)
                    # 첫 정상 준비 자세가 확보되고 세션이 시작되면 첫 반복이 끝날 때까지
                    # 거리(세로 기준값) 검사를 잠근다.
                    if (
                        self.session_active
                        and self._initial_body_height_ratio is not None
                        and not session.completed_reps
                    ):
                        self._distance_locked = True
                    # 준비 자세/반복 종료 전후에만 거리·중심·정면을 검사한다.
                    # 반복 중에는 스쿼트 깊이에 따른 bbox 변화가 카메라 거리로 오인되지 않도록
                    # 필수 관절 visibility만 확인한다.
                    first_rep_locked = self._distance_locked and not session.completed_reps
                    if self._distance_locked and (session.state != "prep" or first_rep_locked):
                        framing = check_pose_visibility(observation.image_landmarks, w, h)
                    else:
                        framing = check_framing(
                            observation.image_landmarks,
                            w,
                            h,
                            self._initial_body_height_ratio,
                        )
                    framing_message = framing.message
                    # 거리 기준은 네 가지 초기 조건(전신·필수 관절·중심·정면)을
                    # 처음 모두 만족한 프레임의 전신 세로 높이로 고정한다.
                    if (
                        self._initial_body_height_ratio is None
                        and self.calibration_active
                        and framing.ok
                        and framing.body_scale is not None
                    ):
                        self._initial_body_height_ratio = framing.body_scale
                    # 카운트다운 전에는 준비 자세 조건을 먼저 만족시켜야 한다.

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

                    if framing_ok and (self.calibration_active or self.session_active):
                        # 카운트다운 중에는 좌표계·지면·준비 자세 기준만 측정하고,
                        # 카운트다운 종료 뒤에만 FSM/DTW 동작 판정을 진행한다.
                        common2d, frozen_mask, mean_conf = bridge.update(observation.image_landmarks, w, h)
                        self._error_feedback_common2d = common2d
                        n_frozen = int(frozen_mask.sum())
                        status = session.push_frame(
                            common2d,
                            evaluate_motion=self.session_active,
                        )
                        self.calibration_ready = bool(
                            session.preparation_calibration_ready
                            and self._initial_body_height_ratio is not None
                        )
                        if (
                            self.session_active
                            and status is not None
                            and status.get("status") == "ok"
                        ):
                            event_name = status["event"]
                            phase = "complete" if event_name == "rep_end" else status["phase"]
                            partial = status["partial_distance"]
                            pelvis_height = status["pelvis_height"]
                            velocity = float(status["velocity"])
                            form_warnings = tuple(status.get("form_warnings", ()))
                            raw_heel_lift_delta = status.get("heel_lift_delta")
                            if raw_heel_lift_delta is not None:
                                heel_lift_delta = tuple(float(value) for value in raw_heel_lift_delta)
                            ground_ready = bool(status.get("ground_ready", False))
                            completion_velocity_ok = abs(velocity) <= session.vel_eps
                            completion_height_ok = (
                                session.baseline_height is not None
                                and pelvis_height >= 0.85 * session.baseline_height
                            )
                            if event_name == "rep_end":
                                completed = status["completed_rep"]
                                if (
                                    completed is not None
                                    and completed.predicted_class in ERROR_VIDEO_LABELS
                                ):
                                    self._error_feedback_class = completed.predicted_class
                                    self._error_feedback_remaining = 90  # 약 3초(30fps)
                                # 완료 프레임에서는 준비 자세 세로 기준과 다시 비교한다.
                                completion_framing = check_framing(
                                    observation.image_landmarks,
                                    w,
                                    h,
                                    self._initial_body_height_ratio,
                                )
                                framing_message = completion_framing.message
                                framing_ok = completion_framing.ok
                                self._distance_locked = False
                            elif status["phase"] != "prep":
                                self._distance_locked = True

                else:
                    video_bgr = display_bgr.copy()
                    skeleton_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.rectangle(video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (60, 60, 240), 2)

                if self._error_feedback_remaining > 0 and self._error_feedback_class is not None:
                    if self._error_feedback_common2d is not None:
                        annotate_error(video_bgr, self._error_feedback_common2d, self._error_feedback_class)
                    cv2.putText(
                        video_bgr,
                        ERROR_VIDEO_LABELS.get(self._error_feedback_class, "ERROR: POSTURE"),
                        (12, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (60, 60, 240), 2, cv2.LINE_AA,
                    )
                    self._error_feedback_remaining -= 1
                    if self._error_feedback_remaining == 0:
                        self._error_feedback_class = None

                # 상승→종료 판정 조건을 좌측 상단에 표시한다.
                # 초록색은 조건 충족, 빨간색은 미충족을 의미한다.
                if pose_found and joint_speed is not None:
                    cv2.putText(
                        video_bgr, f"Joint speed: {joint_speed:.4f}",
                        (12, h - 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 220, 80), 1, cv2.LINE_AA,
                    )
                if pelvis_height is not None:
                    cv2.putText(
                        video_bgr, f"Pelvis H/V: {pelvis_height:.3f}/{velocity:.4f}",
                        (12, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 220, 80), 1, cv2.LINE_AA,
                    )
                if heel_lift_delta is not None:
                    heel_lift_color = (60, 60, 240) if any(
                        warning.startswith("HEEL LIFT") for warning in form_warnings
                    ) else (255, 220, 80)
                    cv2.putText(
                        video_bgr,
                        f"Heel lift L/R: {heel_lift_delta[0]:.3f}/{heel_lift_delta[1]:.3f}",
                        (12, h - 114), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        heel_lift_color, 1, cv2.LINE_AA,
                    )
                for warning_index, warning in enumerate(form_warnings):
                    cv2.putText(
                        video_bgr, warning, (12, 154 + 24 * warning_index),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 240), 2, cv2.LINE_AA,
                    )
                condition_color = lambda ok: (60, 210, 80) if ok else (60, 60, 240)
                cv2.putText(
                    video_bgr, f"End speed: {'OK' if completion_velocity_ok else 'NG'}",
                    (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    condition_color(completion_velocity_ok), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    video_bgr, f"Height return: {'OK' if completion_height_ok else 'NG'}",
                    (12, 101), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    condition_color(completion_height_ok), 2, cv2.LINE_AA,
                )

                self.status_ready.emit(
                    PipelineStatus(
                        video_bgr=video_bgr,
                        skeleton_bgr=skeleton_bgr,
                        fps=smoothed_fps,
                        pose_found=pose_found,
                        mean_confidence=mean_conf,
                        n_frozen=n_frozen,
                        framing_ok=framing_ok,
                        framing_message=framing_message,
                        phase=phase,
                        pelvis_height=pelvis_height,
                        rep_count=len(session.completed_reps),
                        partial_distance=partial,
                        completed_rep=completed,
                        form_warnings=form_warnings,
                        heel_lift_delta=heel_lift_delta,
                        ground_ready=ground_ready,
                        completion_velocity_ok=completion_velocity_ok,
                        completion_height_ok=completion_height_ok,
                        skeleton_world=skeleton_world,
                    )
                )
        except (PoseBackendError, RuntimeError, ValueError, OSError) as error:
            self.fatal_error.emit(str(error))
        except Exception as error:  # noqa: BLE001
            self.fatal_error.emit(f"파이프라인 처리 중 예기치 않은 오류: {error}")
        finally:
            if capture is not None:
                capture.release()
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass


__all__ = ["SquatPipelineWorker", "PipelineStatus"]
