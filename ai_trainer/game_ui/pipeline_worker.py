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
import gc
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PyQt5.QtCore import QThread, pyqtSignal

from ai_trainer.live_pose.mediapipe_pose import MediaPipePoseDetector, PoseBackendError
from ai_trainer.live_pose.worker import CameraConfig, _open_camera
from ai_trainer.lifting_model import TemporalLiftingNet
from ai_trainer.online_dtw import OnlineSquatSession
from ai_trainer.reference_db_io import load_reference_db
from ai_trainer.common_skeleton import COMMON_BONE_COLORS_BGR, COMMON_BONE_INDEX_PAIRS
from ai_trainer.render import draw_skeleton_panel, fit_transform
from ai_trainer.rep_diagnostics import RepDiagnosticRecorder
from ai_trainer.rep_phase_display import rep_phase_label
from ai_trainer.two_d_diagnostic import TwoDDiagnostic
from ai_trainer.posture_score import PostureScorer, RealtimePostureMatcher
from ai_trainer.angle_domain_diagnostic import AngleDomainDiagnostic

from .framing_check import check_active_tracking, check_framing, guide_box as compute_guide_box
from .cv_text import safe_put_text

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "pose_landmarker_full.task"
LIFTING_CKPT = ROOT / "output" / "lifting_baseline" / "model_best.pt"
WEIGHTS_CFG_PATH = ROOT / "configs" / "dtw_feature_weights.json"
DB_DIR = ROOT / "output" / "reference_db"
REP_DIAGNOSTIC_DIR = ROOT / "output" / "rep_diagnostics"
OFFLINE_REPORT_PATH = ROOT / "output" / "dtw_eval" / "offline_eval_report.json"
LIVE_PANEL_W, LIVE_PANEL_H = 640, 480


def _env_enabled(name: str) -> bool:
    """Return True only for an explicit opt-in environment value."""
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _empty_live_skeleton() -> np.ndarray:
    canvas = np.full((LIVE_PANEL_H, LIVE_PANEL_W, 3), 24, dtype=np.uint8)
    import cv2
    text = "Pose not detected"
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
    safe_put_text(
        canvas, text, ((LIVE_PANEL_W - size[0]) // 2, LIVE_PANEL_H // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (154, 165, 181), 1,
        source="live_pose_missing",
    )
    return canvas


def _render_live_skeleton(common2d: np.ndarray) -> np.ndarray:
    """Render the current MediaPipe-derived Common Skeleton on its own canvas."""
    canvas = np.full((LIVE_PANEL_H, LIVE_PANEL_W, 3), 24, dtype=np.uint8)
    points = np.asarray(common2d, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1)
    if not np.any(valid):
        return _empty_live_skeleton()
    transform = fit_transform(points[valid][None, ...], LIVE_PANEL_W, LIVE_PANEL_H, margin=48, flip_y=False)
    fitted = np.full_like(points, np.nan)
    fitted[valid] = transform(points[valid])
    draw_skeleton_panel(
        canvas, (0, 0), LIVE_PANEL_W, LIVE_PANEL_H, fitted,
        "Live MediaPipe 2D", None,
        COMMON_BONE_INDEX_PAIRS, COMMON_BONE_COLORS_BGR,
    )
    return canvas


@dataclass(frozen=True)
class PipelineStatus:
    raw_video_bgr: np.ndarray
    live_skeleton_bgr: np.ndarray
    video_bgr: np.ndarray
    fps: float
    pose_found: bool
    mean_confidence: float
    n_frozen: int
    framing_ok: bool
    prep_framing_ok: bool
    tracking_ok: bool
    active: bool
    framing_message: str
    phase: str | None
    pelvis_height: float | None  # 정규화(leg_length 단위) pelvis 높이 -> 레퍼런스 동기화에 사용
    rep_count: int
    partial_distance: dict | None  # {"phase":..., "distance_by_class": {...}}
    completed_rep: object | None  # ai_trainer.online_dtw.RepResult
    realtime_posture_match: dict | None = None
    depth_debug: dict | None = None
    state_debug: dict | None = None
    capture_monotonic: float = 0.0
    processing_finished_monotonic: float = 0.0
    camera_frame_id: int = -1


class SquatPipelineWorker(QThread):
    status_ready = pyqtSignal(object)  # PipelineStatus
    status_changed = pyqtSignal(str)
    fatal_error = pyqtSignal(str)

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH, config: CameraConfig | None = None, parent=None):
        super().__init__(parent)
        self.model_path = Path(model_path).resolve()
        self.config = config or CameraConfig()
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
        self._last_rep_state = None
        self._last_state_debug_frame = -10_000
        self._standing_torso_samples: list[float] = []
        self._performance_rows: list[dict[str, float]] = []
        self._display_rows: list[dict[str, float]] = []
        self._camera_buffer_report: dict = {}

    FRAMING_DEBOUNCE_FRAMES = 5

    def run(self) -> None:
        capture = None
        detector = None
        session = None
        diagnostic = RepDiagnosticRecorder(REP_DIAGNOSTIC_DIR)
        diagnostic_2d = None
        heel_shadow_logger = None
        heel_validation_logger = None
        camera_frame = -1
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
            try:
                from ai_trainer.heel_shadow_validation import HeelShadowValidationLogger
                heel_validation_logger = HeelShadowValidationLogger(ROOT)
                print(f"[HEEL-DIRECT-SHADOW SESSION] {heel_validation_logger.path}", flush=True)
            except Exception as error:
                print(f"[HEEL-DIRECT-SHADOW WARNING] {type(error).__name__}: {error}", flush=True)
            heavy_diagnostics_enabled = _env_enabled("AI_TRAINER_ENABLE_HEAVY_DIAGNOSTICS")
            partial_dtw_enabled = _env_enabled("AI_TRAINER_ENABLE_PARTIAL_DTW")
            print(
                "[RUNTIME RECOVERY] "
                f"heavy_diagnostics={heavy_diagnostics_enabled}, "
                f"partial_dtw={partial_dtw_enabled}",
                flush=True,
            )
            try:
                from scripts.build_reference_db import TL_ZIP, VL_ZIP
                diagnostic_2d = TwoDDiagnostic(ROOT, TL_ZIP, VL_ZIP)
                if not diagnostic_2d.enabled:
                    print(f"[2D DIAGNOSTIC WARNING] {diagnostic_2d.error}", flush=True)
            except Exception as error:
                print(f"[2D DIAGNOSTIC WARNING] {type(error).__name__}: {error}", flush=True)
            if heavy_diagnostics_enabled:
                try:
                    from ai_trainer.live_heel_shadow import LiveHeelShadowLogger
                    heel_shadow_logger = LiveHeelShadowLogger(ROOT)
                    print(f"[HEEL-SHADOW SESSION] {heel_shadow_logger.path}", flush=True)
                except Exception as error:
                    print(f"[HEEL-SHADOW WARNING] {type(error).__name__}: {error}", flush=True)
            else:
                print("[HEEL-SHADOW] disabled (heavy diagnostics opt-in required)", flush=True)
            score_calib = None
            if OFFLINE_REPORT_PATH.exists():
                score_calib = json.loads(OFFLINE_REPORT_PATH.read_text(encoding="utf-8"))["score_calibration"]

            session = OnlineSquatSession(
                model=lifting_model, device=device, db_operational=db["operational"],
                weights_cfg=weights_cfg, score_calib=score_calib, diagnostic=diagnostic,
                diagnostic_2d=diagnostic_2d if heavy_diagnostics_enabled else None,
                heel_shadow_logger=heel_shadow_logger,
                heel_validation_logger=heel_validation_logger,
                enable_heavy_diagnostics=heavy_diagnostics_enabled,
                enable_partial_dtw=partial_dtw_enabled,
                posture_scorer=PostureScorer(
                    db["operational"].get("정상", []),
                    diagnostic_2d.refs.get("정상", []) if diagnostic_2d is not None and diagnostic_2d.enabled else [],
                    ROOT / "output" / "diagnostics",
                    enable_ab_diagnostic=heavy_diagnostics_enabled,
                ),
                realtime_posture_matcher=RealtimePostureMatcher(
                    diagnostic_2d.refs.get("정상", [])
                    if diagnostic_2d is not None and diagnostic_2d.enabled else []
                ),
                angle_domain_diagnostic=(
                    AngleDomainDiagnostic(
                        diagnostic_2d.refs.get("정상", []),
                        db["operational"].get("정상", []),
                        ROOT / "output" / "diagnostics",
                        weights_cfg.get("2d_semantic_validation", {}),
                    )
                    if (
                        heavy_diagnostics_enabled
                        and diagnostic_2d is not None and diagnostic_2d.enabled
                        and os.environ.get("AI_TRAINER_ANGLE_DOMAIN_DIAGNOSTIC", "0") == "1"
                    ) else None
                ),
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
            self._camera_buffer_report = dict(getattr(_open_camera, "last_report", {}) or {})
            print("\n========== CAMERA ==========", flush=True)
            for key, value in self._camera_buffer_report.items():
                print(f"{key}: {value}", flush=True)
            print("============================\n", flush=True)
            self.status_changed.emit("실행 중")

            previous_time = time.perf_counter()
            smoothed_fps = 0.0
            consecutive_failures = 0

            while not self.isInterruptionRequested():
                frame_started = time.perf_counter()
                camera_frame += 1
                success, frame_bgr = capture.read()
                capture_finished = time.perf_counter()
                if not success or frame_bgr is None:
                    consecutive_failures += 1
                    if consecutive_failures >= 30:
                        raise RuntimeError("카메라 프레임을 연속으로 읽지 못했습니다.")
                    self.msleep(10)
                    continue
                consecutive_failures = 0

                display_bgr = np.ascontiguousarray(frame_bgr[:, ::-1] if self.config.mirror else frame_bgr)
                raw_video_bgr = display_bgr.copy()
                observation = detector.process(np.ascontiguousarray(display_bgr[:, :, ::-1]))
                pose_finished = time.perf_counter()

                now = time.perf_counter()
                instantaneous_fps = 1.0 / max(now - previous_time, 1e-6)
                previous_time = now
                smoothed_fps = instantaneous_fps if smoothed_fps == 0.0 else smoothed_fps * 0.90 + instantaneous_fps * 0.10

                phase = None
                partial = None
                completed = None
                pelvis_height = None
                mean_conf = 0.0
                n_frozen = 0
                depth_debug = None
                state_debug = None
                realtime_posture_match = None
                session_ms = 0.0
                pose_found = observation is not None
                framing_ok = False
                prep_framing_ok = False
                tracking_ok = False
                framing_message = "카메라 앞에 서주세요"
                h, w = display_bgr.shape[:2]
                gbox = compute_guide_box(w, h)

                if pose_found:
                    landmarks = observation.image_landmarks
                    left_hip_y, right_hip_y = float(landmarks[23, 1]), float(landmarks[24, 1])
                    left_knee_y, right_knee_y = float(landmarks[25, 1]), float(landmarks[26, 1])
                    hip_y = (left_hip_y + right_hip_y) / 2.0
                    knee_y = (left_knee_y + right_knee_y) / 2.0
                    depth_debug = {
                        "left_hip_y": left_hip_y,
                        "right_hip_y": right_hip_y,
                        "hip_y": hip_y,
                        "left_knee_y": left_knee_y,
                        "right_knee_y": right_knee_y,
                        "knee_y": knee_y,
                        "hip_knee_diff": hip_y - knee_y,
                    }
                    framing = check_framing(observation.image_landmarks, w, h)
                    prep_framing_ok = framing.ok

                    if not self.session_active and framing.ok and framing.torso_scale is not None:
                        self._standing_torso_samples.append(float(framing.torso_scale))
                        self._standing_torso_samples = self._standing_torso_samples[-30:]
                    standing_torso = (
                        float(np.median(self._standing_torso_samples))
                        if self._standing_torso_samples else None
                    )

                    # 히스테리시스: 판정이 바뀌는 방향으로 연속 N프레임 나와야 실제로 전환.
                    # 한 프레임만 흔들려도 바로 오류로 튀는 것을 막아준다.
                    if self.session_active:
                        active_tracking = check_active_tracking(
                            observation.image_landmarks, w, h, standing_torso
                        )
                        tracking_ok = active_tracking.ok
                        framing_ok = tracking_ok
                        gate = active_tracking
                    else:
                        if framing.ok == self._framing_effective_ok:
                            self._framing_streak = 0
                        else:
                            self._framing_streak += 1
                            if self._framing_streak >= self.FRAMING_DEBOUNCE_FRAMES:
                                self._framing_effective_ok = framing.ok
                                self._framing_streak = 0
                        framing_ok = self._framing_effective_ok
                        tracking_ok = framing_ok
                        gate = framing
                    framing_message = gate.message

                    # LEFT: raw mirrored webcam plus framing guidance only.  Pose bones
                    # intentionally stay off this image; the CENTER panel owns them.
                    box_color = (90, 220, 90) if framing_ok else (60, 60, 240)
                    cv2.rectangle(raw_video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), box_color, 2)
                    if gate.body_box is not None:
                        cv2.rectangle(raw_video_bgr, (gate.body_box[0], gate.body_box[1]), (gate.body_box[2], gate.body_box[3]), box_color, 2)
                    if not framing_ok:
                        safe_put_text(raw_video_bgr, framing.message, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (60, 60, 240), 2, source="position_message")
                        if gate.low_confidence_joints:
                            # 어떤 관절이 구체적으로 안 잡히는지 진단용으로 표시
                            joints_str = ", ".join(f"{name}({v:.2f})" for name, v in gate.low_confidence_joints)
                            safe_put_text(raw_video_bgr, f"인식 약함: {joints_str}", (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 180, 255), 1, source="pose_quality_warning")

                    # One bridge update feeds both visualization and the existing
                    # analysis pipeline.  No additional MediaPipe inference occurs.
                    bridge_started = time.perf_counter()
                    common2d, frozen_mask, mean_conf = bridge.update(observation.image_landmarks, w, h)
                    bridge_finished = time.perf_counter()
                    n_frozen = int(frozen_mask.sum())
                    center_render_started = time.perf_counter()
                    live_skeleton_bgr = _render_live_skeleton(common2d)
                    center_render_finished = time.perf_counter()

                    if framing_ok and self.session_active:
                        # 화각/거리/정면 여부가 학습 데이터(AI Hub camera1)와 맞고, 3-2-1 카운트다운이
                        # 끝나 세션이 명시적으로 시작된 뒤에만 phase/DTW 파이프라인을 진행한다.
                        # 그렇지 않으면 잘못된 프레임이나 아직 자리를 잡는 중인 프레임이 session의
                        # 캘리브레이션/phase 상태기계에 섞여 들어가지 않도록 건너뛴다.
                        visibility = {
                            name: float(landmarks[index, 3])
                            for name, index in {
                                "LHip": 23, "RHip": 24, "LKnee": 25, "RKnee": 26,
                                "LAnkle": 27, "RAnkle": 28, "LHeel": 29, "RHeel": 30,
                                "LFootIndex": 31, "RFootIndex": 32,
                            }.items()
                        }
                        session_started = time.perf_counter()
                        status = session.push_frame(
                            common2d,
                            landmark_visibility=visibility,
                            diagnostic_context={
                                "framing_phase": "ACTIVE",
                                "bbox_height": gate.bbox_height,
                                "bbox_width": gate.bbox_width,
                                "standing_baseline_scale": standing_torso,
                                "framing_ok": bool(prep_framing_ok),
                                "tracking_ok": True,
                                "position_status": gate.message,
                                "frame_delivered": True,
                                "camera_frame": camera_frame,
                            },
                        )
                        session_ms = (time.perf_counter() - session_started) * 1000.0
                        if status is not None and status.get("status") == "ok":
                            phase = status["phase"]
                            partial = status["partial_distance"]
                            pelvis_height = status["pelvis_height"]
                            depth_debug.update(status.get("depth_debug") or {})
                            state_debug = status.get("state_debug")
                            realtime_posture_match = status.get("realtime_posture_match")
                            if state_debug is not None:
                                if not state_debug.get("calibration_ready", False):
                                    framing_message = "준비 자세를 유지해주세요"
                                current_state = state_debug["current_state"]
                                state_changed = current_state != self._last_rep_state
                                if state_changed and self._last_rep_state is not None:
                                    print(
                                        f"[REP PHASE] {rep_phase_label(self._last_rep_state)} → "
                                        f"{rep_phase_label(current_state)}",
                                        flush=True,
                                    )
                                if state_changed and current_state == "bottom":
                                    _print_bottom_pose_debug(depth_debug)
                                stalled = (
                                    not state_debug["condition_passed"]
                                    and state_debug["frame"] - self._last_state_debug_frame >= 90
                                )
                                if state_changed or stalled or state_debug.get("event") is not None:
                                    _print_rep_state_debug(state_debug, partial)
                                    self._last_state_debug_frame = state_debug["frame"]
                                self._last_rep_state = current_state
                            completed = status.get("completed_rep")
                            if completed is not None:
                                _print_rep_debug(completed)
                                print("\n========== ANALYSIS LOG TRACE ==========", flush=True)
                                print("pipeline final signal emitted: True", flush=True)
                                print(f"pipeline completed REP count: {len(session.completed_reps)}", flush=True)
                                print("========================================\n", flush=True)
                            elif status.get("score_ready_rep") is not None:
                                completed = status["score_ready_rep"]
                                _print_rep_debug(completed)

                            if partial is not None:
                                best_class = partial.get("predicted_class")
                                if best_class is not None and best_class != "정상":
                                    # 어떤 오류유형에 가장 가까운지에 따라 관련 관절 옆에 말풍선 설명을 붙인다
                                    # (예: 고관절오류 -> 고관절/상체 근처, claude.md 9장 오류유형별 feature 참고).
                                    pass
                    elif self.session_active:
                        diagnostic.record_framing_skip(
                            camera_frame=camera_frame,
                            reason="FRAMING_NOT_OK",
                            position_state=gate.message,
                            framing_phase="ACTIVE",
                            tracking_ok=False,
                            bbox_height=gate.bbox_height,
                            bbox_width=gate.bbox_width,
                            standing_baseline_scale=standing_torso,
                        )
                else:
                    cv2.rectangle(raw_video_bgr, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (60, 60, 240), 2)
                    live_skeleton_bgr = _empty_live_skeleton()
                    if self.session_active:
                        diagnostic.record_framing_skip(
                            camera_frame=camera_frame,
                            reason="POSE_NOT_FOUND",
                            position_state=framing_message,
                        )

                # Kept for compatibility with callers that still consume video_bgr.
                video_bgr = raw_video_bgr

                processing_finished = time.perf_counter()
                detector_timing = getattr(detector, "last_timing", {})
                session_timing = getattr(session, "last_frame_timing", {}) if session_ms > 0.0 else {}
                bridge_ms = (
                    (bridge_finished - bridge_started) * 1000.0 if pose_found else 0.0
                )
                center_ms = (
                    (center_render_finished - center_render_started) * 1000.0 if pose_found else 0.0
                )
                row = {
                    "frame_id": camera_frame,
                    "camera_frame_id": camera_frame,
                    "capture_start": frame_started,
                    "capture_end": capture_finished,
                    "capture_ms": (capture_finished - frame_started) * 1000.0,
                    "mediapipe_start": capture_finished,
                    "mediapipe_end": pose_finished,
                    "mediapipe_ms": (pose_finished - capture_finished) * 1000.0,
                    "mp_image_conversion_ms": float(detector_timing.get("conversion_ms", 0.0)),
                    "mp_inference_ms": float(detector_timing.get("inference_ms", 0.0)),
                    "mp_postprocess_ms": float(detector_timing.get("postprocess_ms", 0.0)),
                    "common_skeleton_ms": bridge_ms,
                    "lifting_ms": float(session_timing.get("lifting_ms", 0.0)),
                    "partial_dtw_ms": float(session_timing.get("partial_dtw_ms", 0.0)),
                    "realtime_match_ms": float(session_timing.get("realtime_match_ms", 0.0)),
                    "realtime_match_phase": (
                        realtime_posture_match.get("phase") if realtime_posture_match else None
                    ),
                    "realtime_match_valid": bool(
                        realtime_posture_match and realtime_posture_match.get("match_valid")
                    ),
                    "realtime_match_percent": (
                        realtime_posture_match.get("match_percent")
                        if realtime_posture_match else None
                    ),
                    "realtime_match_raw": (
                        realtime_posture_match.get("raw_match")
                        if realtime_posture_match else None
                    ),
                    "rep_detector_ms": float(session_timing.get("rep_detector_ms", 0.0)),
                    "semantic_ms": float(session_timing.get("semantic_ms", 0.0)),
                    "final_dtw_ms": float(session_timing.get("final_dtw_ms", 0.0)),
                    "render_left_ms": max(0.0, (bridge_started - pose_finished) * 1000.0) if pose_found else 0.0,
                    "render_center_ms": center_ms,
                    "render_right_ms": None,
                    "session_ms": session_ms,
                    "other_render_ms": (processing_finished - pose_finished) * 1000.0 - session_ms,
                    "worker_total_ms": (processing_finished - frame_started) * 1000.0,
                    "total_ms": (processing_finished - frame_started) * 1000.0,
                    "capture_time": capture_finished,
                    "processing_finished": processing_finished,
                    "ui_display_ms": None,
                    "frame_age_ms": None,
                    "ui_frame_lag": None,
                    "diagnostic_active": bool(
                        session is not None and (
                            any(not item.done() for item in session.diagnostic_futures)
                            or any(not item.done() for item in session.score_futures)
                            or any(not item.done() for item in session.final_analysis_futures)
                        )
                    ),
                    "gc_collections": sum(item["collections"] for item in gc.get_stats()),
                }
                self._performance_rows.append(row)
                self._performance_rows = self._performance_rows[-1800:]
                self.status_ready.emit(
                    PipelineStatus(
                        raw_video_bgr=raw_video_bgr,
                        live_skeleton_bgr=live_skeleton_bgr,
                        video_bgr=video_bgr,
                        fps=smoothed_fps,
                        pose_found=pose_found,
                        mean_confidence=mean_conf,
                        n_frozen=n_frozen,
                        framing_ok=framing_ok,
                        prep_framing_ok=prep_framing_ok,
                        tracking_ok=tracking_ok,
                        active=self.session_active,
                        framing_message=framing_message,
                        phase=phase,
                        pelvis_height=pelvis_height,
                        rep_count=session.submitted_rep_count,
                        partial_distance=partial,
                        completed_rep=completed,
                        realtime_posture_match=realtime_posture_match,
                        depth_debug=depth_debug,
                        state_debug=state_debug,
                        capture_monotonic=capture_finished,
                        processing_finished_monotonic=processing_finished,
                        camera_frame_id=camera_frame,
                    )
                )
        except (PoseBackendError, RuntimeError, ValueError, OSError) as error:
            self.fatal_error.emit(str(error))
        except Exception as error:  # noqa: BLE001
            self.fatal_error.emit(f"파이프라인 처리 중 예기치 않은 오류: {error}")
        finally:
            diagnostic.close()
            if session is not None:
                session.close_final_analysis()
                session.close_scoring()
                session.close_diagnostics()
                if session.posture_scorer is not None:
                    session.posture_scorer.close()
                if session.angle_domain_diagnostic is not None:
                    session.angle_domain_diagnostic.close()
            if diagnostic_2d is not None:
                diagnostic_2d.close()
            if heel_shadow_logger is not None:
                heel_shadow_logger.close()
            if heel_validation_logger is not None:
                try:
                    heel_validation_logger.close()
                except Exception as error:
                    print(f"[HEEL-DIRECT-SHADOW WARNING] close failed: {type(error).__name__}: {error}", flush=True)
            if capture is not None:
                capture.release()
            if detector is not None:
                try:
                    detector.close()
                except Exception:
                    pass
            self._write_performance_summary(session)

    def record_display(self, frame_id: int, capture_time: float, ui_ms: float, right_render_ms: float) -> None:
        now = time.perf_counter()
        self._display_rows.append({
            "frame_id": frame_id,
            "display_time": now,
            "frame_age_ms": max(0.0, (now - capture_time) * 1000.0),
            "ui_ms": ui_ms,
        })
        for row in reversed(self._performance_rows):
            if row["frame_id"] == frame_id:
                row["ui_display_ms"] = ui_ms
                row["frame_age_ms"] = max(0.0, (now - capture_time) * 1000.0)
                row["render_right_ms"] = right_render_ms
                row["ui_frame_lag"] = max(0, self._performance_rows[-1]["frame_id"] - frame_id)
                break
        self._display_rows = self._display_rows[-1800:]

    def _write_performance_summary(self, session: OnlineSquatSession | None) -> None:
        def stats(values):
            values = np.asarray(values, dtype=float)
            if not len(values):
                return {"count": 0, "avg": None, "p95": None, "max": None}
            return {"count": int(len(values)), "avg": float(np.mean(values)),
                    "p95": float(np.percentile(values, 95)), "max": float(np.max(values))}

        summary = {
            "camera_buffer": self._camera_buffer_report,
            "stages_ms": {
                key: stats([row[key] for row in self._performance_rows])
                for key in (
                    "capture_ms", "mediapipe_ms", "session_ms", "realtime_match_ms",
                    "other_render_ms", "total_ms",
                )
            },
            "capture_interval_ms": stats(np.diff([row["capture_time"] for row in self._performance_rows]) * 1000.0),
            "display_interval_ms": stats(np.diff([row["display_time"] for row in self._display_rows]) * 1000.0),
            "frame_age_ms": stats([row["frame_age_ms"] for row in self._display_rows]),
            "ui_ms": stats([row["ui_ms"] for row in self._display_rows]),
            "post_rep_diagnostic_ms": stats(session.diagnostic_timings_ms if session is not None else []),
            "completed_posture_match_ms": stats([
                float((getattr(rep, "posture_score", None) or {}).get("calculation_ms", 0.0))
                for rep in (session.completed_reps if session is not None else [])
                if (getattr(rep, "posture_score", None) or {}).get("score_valid")
            ]),
            "final_analysis_ms": stats([
                row["analysis_ms"] for row in (session.final_analysis_timings if session is not None else [])
            ]),
            "final_analysis_queue_wait_ms": stats([
                row["queue_wait_ms"] for row in (session.final_analysis_timings if session is not None else [])
            ]),
            "final_dtw_background_ms": stats([
                row["final_dtw_ms"] for row in (session.final_analysis_timings if session is not None else [])
            ]),
            "final_analysis_max_queue_depth": (
                session.final_analysis_max_queue_depth if session is not None else 0
            ),
            "slowest_frames": sorted(
                self._performance_rows, key=lambda row: row["worker_total_ms"], reverse=True
            )[:10],
        }
        out_dir = ROOT / "output" / "diagnostics"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / time.strftime("performance_%Y%m%d_%H%M%S.json")
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        trace_path = out_dir / path.name.replace("performance_", "performance_frames_").replace(".json", ".jsonl")
        trace_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self._performance_rows),
            encoding="utf-8",
        )
        realtime = summary["stages_ms"]["realtime_match_ms"]
        print("\n[REALTIME MATCH PERF]", flush=True)
        print(f"avg: {_fmt_debug(realtime['avg'], 3)} ms", flush=True)
        print(f"P95: {_fmt_debug(realtime['p95'], 3)} ms", flush=True)
        print(f"max: {_fmt_debug(realtime['max'], 3)} ms", flush=True)
        print(f"[PERFORMANCE SUMMARY] {path}", flush=True)


__all__ = ["SquatPipelineWorker", "PipelineStatus"]


def _fmt_debug(value, digits: int = 6) -> str:
    if value is None:
        return "None"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _print_rep_debug(result: object) -> None:
    debug = getattr(result, "debug_summary", None)
    if not debug:
        return
    distances = debug["class_distances"]
    print("\n========== REP DEBUG ==========", flush=True)
    print(f"REP: {result.rep_index + 1}", flush=True)
    print(f"DTW raw predicted class: {debug['raw_dtw_predicted_class']}", flush=True)
    print(f"DTW final class: {debug['final_predicted_class']}", flush=True)
    for class_label, distance in sorted(distances.items(), key=lambda item: item[1]):
        print(f"DTW distance [{class_label}]: {_fmt_debug(distance)}", flush=True)
    print(f"Nearest-class margin: {_fmt_debug(debug['nearest_margin'])}", flush=True)
    print(f"Reference depth threshold: {_fmt_debug(debug['normal_depth_threshold'])}", flush=True)
    print(f"User standing pelvis: {_fmt_debug(debug['standing_pelvis_height'])}", flush=True)
    print(f"User tracked minimum pelvis: {_fmt_debug(debug['tracked_min_pelvis_height'])}", flush=True)
    print(f"User recomputed minimum pelvis: {_fmt_debug(debug['recomputed_min_pelvis_height'])}", flush=True)
    print(f"Depth difference (user-threshold): {_fmt_debug(debug['depth_difference'])}", flush=True)
    print(f"Bottom hip angle: {_fmt_debug(debug['bottom_hip_angle'], 2)} deg", flush=True)
    print(f"Bottom knee angle: {_fmt_debug(debug['bottom_knee_angle'], 2)} deg", flush=True)
    print(f"2D knee excursion: {_fmt_debug(debug.get('knee_excursion_2d'), 2)} deg", flush=True)
    print(f"2D hip excursion: {_fmt_debug(debug.get('hip_excursion_2d'), 2)} deg", flush=True)
    heel = debug.get("heel_semantic") or {}
    if debug.get("raw_dtw_predicted_class") == "발뒤꿈치오류":
        print(
            "Heel semantic: "
            f"{heel.get('verdict', '-')} value={_fmt_debug(heel.get('value'))} "
            f"range=({_fmt_debug(heel.get('no_max'))}, {_fmt_debug(heel.get('yes_min'))})",
            flush=True,
        )
    print(f"Reason: {debug.get('reason_code', '-')}", flush=True)
    posture = getattr(result, "posture_score", None) or {}
    print("\n[POSTURE MATCH]", flush=True)
    print(f"REP: {result.rep_index + 1}", flush=True)
    print(f"match_valid: {posture.get('score_valid')}", flush=True)
    print(f"overall_match: {_fmt_debug(posture.get('overall'), 1)}%", flush=True)
    for name in ("depth", "hip", "knee", "heel_stability", "balance", "trajectory"):
        print(f"{name}: {_fmt_debug(posture.get(name), 1)}", flush=True)
    print(f"production_raw: {debug.get('raw_dtw_predicted_class')}", flush=True)
    print(f"production_final: {debug.get('final_predicted_class')}", flush=True)
    print(f"classification_unknown: {debug.get('final_predicted_class') == '자세추정불확실'}", flush=True)
    print(f"measurement_valid: {posture.get('measurement_valid')}", flush=True)
    print(f"calculation_ms: {_fmt_debug(posture.get('calculation_ms'), 3)}", flush=True)
    diagnostic = posture.get("diagnostic") or {}
    if diagnostic:
        print("\n[POSTURE MATCH DIAGNOSTIC]", flush=True)
        print(
            f"2D knee excursion: {_fmt_debug(diagnostic.get('knee_excursion_2d'), 2)} | "
            f"normal median: {_fmt_debug(diagnostic.get('knee_excursion_normal_median'), 2)} | "
            f"difference: {_fmt_debug((diagnostic.get('knee_excursion_2d') or 0.0) - (diagnostic.get('knee_excursion_normal_median') or 0.0), 2)}",
            flush=True,
        )
        print(
            f"2D hip excursion: {_fmt_debug(diagnostic.get('hip_excursion_2d'), 2)} | "
            f"normal median: {_fmt_debug(diagnostic.get('hip_excursion_normal_median'), 2)} | "
            f"difference: {_fmt_debug((diagnostic.get('hip_excursion_2d') or 0.0) - (diagnostic.get('hip_excursion_normal_median') or 0.0), 2)}",
            flush=True,
        )
        for name, values in (diagnostic.get("component_distances") or {}).items():
            print(
                f"{name}: distance={_fmt_debug(values.get('live_distance'))} | "
                f"normal calibration={_fmt_debug(values.get('normal_calibration_distance'))} | "
                f"ratio={_fmt_debug(values.get('distance_ratio'), 2)}",
                flush=True,
            )
        print(f"raw match: {_fmt_debug(diagnostic.get('raw_match'), 1)}%", flush=True)
        print(f"mapped match: {_fmt_debug(diagnostic.get('mapped_match'), 1)}%", flush=True)
    print("================================\n", flush=True)
    return  # Detailed traces remain in output diagnostics instead of blocking the live console.
    print("\n[2D / 3D Consistency]", flush=True)
    for joint in ("knee", "hip"):
        title = joint.capitalize()
        print(
            f"{title} 2D standing/bottom/excursion: "
            f"{_fmt_debug(debug.get(f'standing_{joint}_angle_2d'), 2)} / "
            f"{_fmt_debug(debug.get(f'bottom_{joint}_angle_2d'), 2)} / "
            f"{_fmt_debug(debug.get(f'{joint}_excursion_2d'), 2)} deg",
            flush=True,
        )
        print(
            f"{title} 3D standing/bottom/excursion: "
            f"{_fmt_debug(debug.get(f'standing_{joint}_angle_3d'), 2)} / "
            f"{_fmt_debug(debug.get(f'bottom_{joint}_angle_3d'), 2)} / "
            f"{_fmt_debug(debug.get(f'{joint}_excursion_3d'), 2)} deg",
            flush=True,
        )
        print(
            f"{title} difference/range/result: {_fmt_debug(debug.get(f'{joint}_consistency_delta'), 2)} / "
            f"{debug.get(f'{joint}_consistency_range')} / "
            f"{'PASS' if debug.get(f'{joint}_consistency_pass') else 'FAIL'}",
            flush=True,
        )
    print(f"Final consistency: {'PASS' if debug.get('consistency_pass') else 'FAIL'}", flush=True)
    print(
        f"Temporal alignment current/best offset: {debug.get('temporal_alignment_current_offset')} / "
        f"{debug.get('temporal_alignment_best_offset')}", flush=True,
    )
    for row in debug.get("temporal_alignment_offsets", []):
        print(
            f"  offset {row['offset']:+d}: 2D knee={row['knee_2d']:.2f}, "
            f"2D hip={row['hip_2d']:.2f}, angle error={row['angle_error']:.2f}",
            flush=True,
        )
    print(f"2D semantic depth: {'PASS' if debug.get('depth_2d_pass') else 'FAIL'}", flush=True)
    print(f"3D consistency: {'PASS' if debug.get('consistency_pass') else 'FAIL'}", flush=True)
    print(f"Input quality: {debug.get('input_quality', '-')}", flush=True)
    _print_bottom_pose_debug(debug.get("bottom_pose"), heading="REP LOWEST POSE DEBUG")
    print(f"All values finite: {debug['values_finite']}", flush=True)
    print(f"Depth result: {'NORMAL DEPTH' if debug['depth_pass'] else 'INSUFFICIENT DEPTH'}", flush=True)
    print(f"Final result: {debug['final_predicted_class']}", flush=True)
    print(f"Reason: {debug['reason']}", flush=True)
    print("\n[Normal References]", flush=True)
    for ref in debug["normal_references"]:
        print(
            f"{ref['id']}: pelvis range={_fmt_debug(ref['pelvis_min'])}..{_fmt_debug(ref['pelvis_max'])}, "
            f"standing={_fmt_debug(ref['standing_pelvis'])}, bottom_frame={ref['bottom_frame']}, "
            f"bottom_hip={_fmt_debug(ref['bottom_hip_angle'], 2)}, "
            f"bottom_knee={_fmt_debug(ref['bottom_knee_angle'], 2)}",
            flush=True,
        )
    print("\n[Hip-down-error References]", flush=True)
    for ref in debug["hip_down_references"]:
        print(
            f"{ref['id']}: pelvis range={_fmt_debug(ref['pelvis_min'])}..{_fmt_debug(ref['pelvis_max'])}, "
            f"standing={_fmt_debug(ref['standing_pelvis'])}, bottom_frame={ref['bottom_frame']}, "
            f"bottom_hip={_fmt_debug(ref['bottom_hip_angle'], 2)}, "
            f"bottom_knee={_fmt_debug(ref['bottom_knee_angle'], 2)}",
            flush=True,
        )
    print("\n[User REP Frame Trace]", flush=True)
    print("stream_frame,rep_frame,phase,pelvis_height,running_min,hip_angle,knee_angle", flush=True)
    for row in result.debug_trace:
        print(
            f"{row['frame']},{row['rep_frame']},{row['phase']},"
            f"{_fmt_debug(row['pelvis_height'])},{_fmt_debug(row['rep_min_pelvis_height'])},"
            f"{_fmt_debug(row['hip_angle'], 2)},{_fmt_debug(row['knee_angle'], 2)}",
            flush=True,
        )
    print("===============================\n", flush=True)


def _print_bottom_pose_debug(debug: dict | None, heading: str = "BOTTOM POSE DEBUG") -> None:
    if not debug:
        return
    landmarks = debug.get("landmarks_2d", {})
    print(f"\n========== {heading} ==========", flush=True)
    print(
        "2D hip angle L/R/avg: "
        f"{_fmt_debug(debug.get('left_hip_angle_2d'), 2)} / "
        f"{_fmt_debug(debug.get('right_hip_angle_2d'), 2)} / "
        f"{_fmt_debug(debug.get('hip_angle_2d'), 2)} deg",
        flush=True,
    )
    print(
        "2D knee angle L/R/avg: "
        f"{_fmt_debug(debug.get('left_knee_angle_2d'), 2)} / "
        f"{_fmt_debug(debug.get('right_knee_angle_2d'), 2)} / "
        f"{_fmt_debug(debug.get('knee_angle_2d'), 2)} deg",
        flush=True,
    )
    print(
        "3D hip angle L/R/avg: "
        f"{_fmt_debug(debug.get('left_hip_angle_3d'), 2)} / "
        f"{_fmt_debug(debug.get('right_hip_angle_3d'), 2)} / "
        f"{_fmt_debug(debug.get('hip_angle_3d'), 2)} deg",
        flush=True,
    )
    print(
        "3D knee angle L/R/avg: "
        f"{_fmt_debug(debug.get('left_knee_angle_3d'), 2)} / "
        f"{_fmt_debug(debug.get('right_knee_angle_3d'), 2)} / "
        f"{_fmt_debug(debug.get('knee_angle_3d'), 2)} deg",
        flush=True,
    )
    print(f"Pelvis height: {_fmt_debug(debug.get('pelvis_height'))}", flush=True)
    for joint in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"):
        print(f"{joint} 2D landmark: {landmarks.get(joint, '-')}", flush=True)
    visibility = debug.get("visibility", {})
    if visibility:
        print("MediaPipe visibility:", flush=True)
        for name in ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle", "LHeel", "RHeel", "LFootIndex", "RFootIndex"):
            print(f"  {name}: {_fmt_debug(visibility.get(name), 3)}", flush=True)
    root_centered = debug.get("root_centered_2d", {})
    if root_centered:
        print("Root-centered 2D coordinates:", flush=True)
        for name in ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle"):
            point = root_centered.get(name)
            if point is not None:
                print(f"  {name}: ({point[0]:.3f}, {point[1]:.3f})", flush=True)
    print("=======================================\n", flush=True)


def _print_rep_state_debug(state: dict, partial: dict | None) -> None:
    distances = partial.get("distance_by_class", {}) if partial else {}
    normal_distance = distances.get("정상")
    hip_down_distance = distances.get("엉덩이하방오류")
    difference = (
        normal_distance - hip_down_distance
        if normal_distance is not None and hip_down_distance is not None
        else None
    )
    print("\n========== REP STATE ==========", flush=True)
    print(f"previous phase: {state['previous_state']}", flush=True)
    print(f"current phase: {state['current_state']}", flush=True)
    print(f"next phase: {state['next_state']}", flush=True)
    print(f"event: {state['event']}", flush=True)
    print(f"pelvis height: {_fmt_debug(state['pelvis_height'])}", flush=True)
    print(f"pelvis velocity: {_fmt_debug(state['pelvis_velocity'])}", flush=True)
    print(f"baseline height: {_fmt_debug(state['baseline_height'])}", flush=True)
    print(f"return height threshold: {_fmt_debug(state['return_height_threshold'])}", flush=True)
    print(f"transition condition: {state['reason']}", flush=True)
    print(f"actual value: {_fmt_debug(state['actual'])}", flush=True)
    print(f"required: {state['required']}", flush=True)
    print(f"condition passed: {state['condition_passed']}", flush=True)
    print(f"debounce: {state['debounce_count']} / {state['debounce_required']}", flush=True)
    print(f"NORMAL DTW: {_fmt_debug(normal_distance)}", flush=True)
    print(f"HIP-DOWN DTW: {_fmt_debug(hip_down_distance)}", flush=True)
    print(f"NORMAL - HIP-DOWN: {_fmt_debug(difference)}", flush=True)
    if distances:
        ordered = sorted(distances.items(), key=lambda item: item[1])
        for class_label, distance in ordered:
            print(f"PARTIAL DTW [{class_label}]: {_fmt_debug(distance)}", flush=True)
        margin = ordered[1][1] - ordered[0][1] if len(ordered) > 1 else None
        print(f"RAW best: {partial.get('raw_best_class')}", flush=True)
        print(f"Validated candidate: {partial.get('validated_class')}", flush=True)
        print("Displayed verdict: UNKNOWN / analysis in progress", flush=True)
        print(f"Best-vs-second margin: {_fmt_debug(margin)}", flush=True)
    print("================================\n", flush=True)
