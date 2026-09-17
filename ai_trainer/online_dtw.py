"""Online(스트리밍) Squat 평가 세션.

prerecorded 시퀀스를 프레임 단위로 `push_frame()`에 순차 입력해 실제 웹캠 스트림처럼
시뮬레이션한다. 핵심 제약: **미래 프레임을 참조하지 않는다.**

지연(lag) 설계
---------------
Lifting 모델은 T=9 대칭 윈도우([c-4, c+4])로 학습되었으므로, 프레임 c의 3D를
확정하려면 c+4까지 스트림에 도착해야 한다. 이는 "미래를 안다"가 아니라 **고정
4프레임 출력 지연**이다 — 프레임 c를 확정하는 시점(스트림 위치 c+4)에서 c+4보다
미래의 데이터는 전혀 사용하지 않는다. 실시간 자막/스트리밍 시스템과 동일한 방식.

캘리브레이션
------------
- Scale(leg_length), Orientation(body-centered 축)은 세션 시작 후 처음
  `calib_frames`개 프레임(서 있는 준비 자세로 가정)만으로 1회 계산해 세션 전체에
  고정 적용한다 — 웹캠이 세션 중 움직이지 않는다는 실사용 가정과 일치하며,
  이 역시 미래 데이터를 쓰지 않는다(초반 캘리브레이션 구간 이후 프레임에는
  과거 계산된 값만 사용).

fps 가정
--------
`vel_eps`/`debounce_n`/`calib_frames`/`_causal_velocity`의 `win` 등은 전부 "프레임 수"
단위 임계값이며, AI Hub 레퍼런스 데이터가 30fps로 캡처되었다는 사실(claude.md 7장,
annotation.json의 start_time/end_time 대비 start_frame/end_frame 역산으로 검증)에
맞춰 조정되었다. 웹캠도 `CameraConfig.requested_fps=30`으로 30fps를 목표로 하므로
프레임 수 기반 임계값이 그대로 맞는다. 실제 처리 속도가 추론 부하 등으로 30fps보다
크게 떨어지면 이 임계값들의 실제 시간 의미가 달라지므로 재보정이 필요할 수 있다.
"""
from __future__ import annotations

import copy
import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from .common_skeleton import COMMON_JOINT_NAMES
from .dtw_compare import PHASES, multi_reference_distance, resolve_weights
from .features import extract_all_features
from .lifting_dataset import WINDOW_T
from .normalization import body_axes, hip_center_3d, leg_length_scale
from .rep_detector_2d import Adaptive2DRepDetector, validate_dtw_candidate
from .rep_phase_display import rep_phase_label
from .baseline_calibration import StandingBaselineCalibrator

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_L_HIP, _R_HIP = _IDX["LHip"], _IDX["RHip"]
_L_KNEE, _R_KNEE = _IDX["LKnee"], _IDX["RKnee"]
_L_ANKLE, _R_ANKLE = _IDX["LAnkle"], _IDX["RAnkle"]
_NECK = _IDX["Neck"]
_VERTICAL_AXIS = 1
_HALF = WINDOW_T // 2  # =4


def validate_production_candidate(raw_pred: str, *, depth_2d_pass: bool,
                                  consistency_pass: bool) -> str:
    """Apply active semantic guards; lifting consistency remains diagnostic-only."""
    _ = consistency_pass
    return validate_dtw_candidate(
        raw_pred, depth_2d_pass=depth_2d_pass, consistency_pass=True
    )


@dataclass
class RepResult:
    rep_index: int
    frame_range: tuple[int, int]
    predicted_class: str
    raw_distance_by_class: dict[str, float]
    score_vs_normal: float | None
    top_contributing_features: list[tuple[str, float]]
    posture_score: dict | None = None
    debug_summary: dict | None = None
    debug_trace: list[dict] = field(default_factory=list)


@dataclass
class OnlineSquatSession:
    model: torch.nn.Module
    device: torch.device
    db_operational: dict[str, list[dict]]  # class -> [{feat,bounds,meta}]
    weights_cfg: dict
    weight_profile: str | None = None  # None -> weights_cfg["default_profile"]
    calib_frames: int = 8
    vel_eps: float = 0.004
    debounce_n: int = 3
    score_calib: dict | None = None
    diagnostic: object | None = None  # best-effort writer; never participates in decisions
    diagnostic_2d: object | None = None  # diagnostic-only; runs after final result exists
    heel_shadow_logger: object | None = None  # file-only shadow policy logger
    heel_validation_logger: object | None = None  # lightweight Patch-3 shadow logger
    posture_scorer: object | None = None
    realtime_posture_matcher: object | None = None
    angle_domain_diagnostic: object | None = None
    enable_heavy_diagnostics: bool = False
    enable_partial_dtw: bool = False

    # --- 내부 상태 (push_frame이 갱신) ---
    raw2d_buffer: list = field(default_factory=list)  # 원본(정규화 전) common-skeleton 2D
    visibility_buffer: list[dict[str, float]] = field(default_factory=list)
    scale2d: float | None = None
    aligned_seq: list = field(default_factory=list)  # 세션 전체, emit된(지연 적용) 정규화 3D
    emit_offset: int | None = None  # aligned_seq[0]에 해당하는 emit_idx(원본 스트림 인덱스)
    pelvis_height_raw_hist: list = field(default_factory=list)
    pelvis_height_hist: list = field(default_factory=list)  # causal median-filtered
    pose_angle_hist: list = field(default_factory=list)  # mean of 2D hip/knee angles
    R_body: np.ndarray | None = None
    baseline_height: float | None = None
    baseline_pose_angle: float | None = None
    baseline_knee_angle_2d: float | None = None
    baseline_hip_angle_2d: float | None = None
    baseline_knee_angle_3d: float | None = None
    baseline_hip_angle_3d: float | None = None
    smoothed_knee_angle_2d: float | None = None
    smoothed_hip_angle_2d: float | None = None
    knee_angle_2d_hist: list = field(default_factory=list)
    hip_angle_2d_hist: list = field(default_factory=list)
    knee_angle_3d_hist: list = field(default_factory=list)
    hip_angle_3d_hist: list = field(default_factory=list)
    baseline_calibrator: StandingBaselineCalibrator | None = None
    baseline_calibration_result: dict | None = None
    last_baseline_candidate: dict | None = None
    rep_detector_2d: Adaptive2DRepDetector | None = None
    rep_detector_mode: str = "adaptive_2d"
    normal_depth_threshold: float | None = None
    rep_min_pelvis_height: float | None = None

    state: str = "prep"  # prep/descend/bottom/ascend
    debounce_ctr: int = 0
    rep_start_idx: int | None = None
    phase_boundaries_running: dict = field(default_factory=dict)  # 현재 rep의 phase별 [start, end)
    current_phase_start: int = 0
    completed_reps: list = field(default_factory=list)
    partial_dtw_last_frame: int | None = None
    partial_dtw_last_state: str | None = None
    partial_dtw_cache: dict | None = None
    diagnostic_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="rep-diagnostic")
    )
    diagnostic_futures: list = field(default_factory=list)
    diagnostic_timings_ms: list[float] = field(default_factory=list)
    score_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="posture-score")
    )
    score_futures: list = field(default_factory=list)
    scored_results: list = field(default_factory=list)
    final_analysis_executor: ThreadPoolExecutor = field(
        default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="final-analysis")
    )
    final_analysis_futures: list = field(default_factory=list)
    final_analysis_results: queue.Queue = field(default_factory=queue.Queue)
    final_analysis_timings: list[dict] = field(default_factory=list)
    submitted_rep_count: int = 0
    final_analysis_max_queue_depth: int = 0

    def __post_init__(self):
        if self.weight_profile is None:
            self.weight_profile = self.weights_cfg.get("default_profile", "E_full_uniform")
        normal_bottom_heights = []
        for medoid in self.db_operational.get("정상", []):
            start, end = medoid["bounds"].get("최저점", (0, 0))
            values = medoid["feat"]["pelvis_trajectory"][start:end, 0]
            if len(values):
                normal_bottom_heights.append(float(np.min(values)))
        if normal_bottom_heights:
            percentile = float(
                self.weights_cfg.get("depth_validation", {}).get(
                    "normal_reference_percentile", 90.0
                )
            )
            self.normal_depth_threshold = float(
                np.percentile(normal_bottom_heights, percentile)
            )
        semantic = self.weights_cfg.get("2d_semantic_validation", {})
        baseline_cfg = self.weights_cfg.get("baseline_calibration", {})
        self.baseline_calibrator = StandingBaselineCalibrator(
            required_samples=int(baseline_cfg.get("required_consecutive_samples", self.calib_frames)),
            hip_min_deg=float(baseline_cfg.get("hip_average_min_deg", 163.47148059)),
            knee_min_deg=float(baseline_cfg.get("knee_average_min_deg", 131.44119491)),
            hip_side_min_deg=float(baseline_cfg.get("hip_side_min_deg", 154.09205630)),
            knee_side_min_deg=float(baseline_cfg.get("knee_side_min_deg", 111.20578404)),
            hip_delta_max_deg=float(baseline_cfg.get("hip_frame_delta_p90_deg", 5.63620138)),
            knee_delta_max_deg=float(baseline_cfg.get("knee_frame_delta_p90_deg", 7.23585562)),
        )
        self.rep_detector_2d = Adaptive2DRepDetector(
            knee_normal_excursion=float(semantic.get("knee_excursion_min_deg", 46.38858745051192)),
            hip_normal_excursion=float(semantic.get("hip_excursion_min_deg", 27.433186772026716)),
            start_fraction=float(semantic.get("detector_start_fraction", 0.20)),
            valid_rep_fraction=float(semantic.get("detector_valid_rep_fraction", 0.35)),
            recovery_fraction=float(semantic.get("detector_recovery_fraction", 0.20)),
            reversal_fraction=float(semantic.get("detector_reversal_fraction", 0.10)),
            transition_frames=int(semantic.get("detector_transition_frames", 2)),
            recovery_frames=int(semantic.get("detector_recovery_frames", 3)),
        )

    # ------------------------------------------------------------------
    def push_frame(
        self, raw2d_frame: np.ndarray, landmark_visibility: dict[str, float] | None = None,
        diagnostic_context: dict | None = None,
    ) -> dict | None:
        """raw2d_frame: (18,2) camera1 common-skeleton pixel 좌표, 이번에 새로 도착한 프레임.

        반환: 이번 호출로 새로 "확정(emit)"된 과거 프레임(있다면)의 상태 dict, 없으면 None.
        """
        frame_timing = {"lifting_ms": 0.0, "partial_dtw_ms": 0.0,
                        "rep_detector_ms": 0.0, "semantic_ms": 0.0,
                        "final_dtw_ms": 0.0, "realtime_match_ms": 0.0}
        self.last_frame_timing = frame_timing
        self.raw2d_buffer.append(raw2d_frame)
        self.visibility_buffer.append(dict(landmark_visibility or {}))
        n = len(self.raw2d_buffer)
        emit_idx = n - 1 - _HALF  # 이 인덱스가 이제 좌우 half씩 문맥을 확보함
        if emit_idx < 0:
            return None  # 아직 워밍업 (초기 half 프레임 부족)

        lo = max(0, emit_idx - _HALF)
        window = np.stack(self.raw2d_buffer[lo : emit_idx + _HALF + 1])
        if window.shape[0] < WINDOW_T:  # 시퀀스 시작부: 왼쪽 부족분 edge-replication
            pad = WINDOW_T - window.shape[0]
            window = np.concatenate([np.repeat(window[:1], pad, axis=0), window], axis=0)

        # --- 2D 정규화: scale은 calib_frames 이후 고정 (causal) ---
        pelvis2d = window[:, _IDX["Hip"], :]
        neck2d = window[:, _IDX["Neck"], :]
        if self.scale2d is None:
            if emit_idx + 1 >= self.calib_frames:
                torso_lens = [
                    float(np.linalg.norm(self.raw2d_buffer[i][_IDX["Neck"]] - self.raw2d_buffer[i][_IDX["Hip"]]))
                    for i in range(min(self.calib_frames, len(self.raw2d_buffer)))
                ]
                self.scale2d = max(float(np.median(torso_lens)), 1e-6)
            else:
                return {"status": "calibrating", "frame": emit_idx}

        centered2d = window - window[:, _IDX["Hip"] : _IDX["Hip"] + 1, :]
        norm2d = centered2d / self.scale2d

        # --- lifting 모델 추론 (center frame만) ---
        x = torch.from_numpy(norm2d[None].astype(np.float32)).to(self.device)
        lifting_started = time.perf_counter()
        with torch.no_grad():
            hip_centered_3d = self.model(x)[0].cpu().numpy()  # (18,3), 이미 Hip-centered
        frame_timing["lifting_ms"] = (time.perf_counter() - lifting_started) * 1000.0

        # --- scale/orientation: calib_frames 구간에서 1회 계산, 이후 고정 ---
        leg_len = float(leg_length_scale(hip_centered_3d[None])[0])
        if self.R_body is None:
            self._calib_buffer_3d = getattr(self, "_calib_buffer_3d", [])
            self._calib_buffer_3d.append((hip_centered_3d, leg_len))
            if len(self._calib_buffer_3d) >= self.calib_frames:
                arr = np.stack([c for c, _ in self._calib_buffer_3d])
                scale0 = float(np.median([s for _, s in self._calib_buffer_3d]))
                scale0 = max(scale0, 1e-6)
                scaled0 = arr / scale0
                axes = np.stack([body_axes(f) for f in scaled0]).mean(axis=0)
                lateral = axes[:, 0] / np.linalg.norm(axes[:, 0])
                forward = np.cross(lateral, axes[:, 1])
                forward /= np.linalg.norm(forward)
                vertical = np.cross(forward, lateral)
                vertical /= np.linalg.norm(vertical)
                self.R_body = np.stack([lateral, vertical, forward], axis=1)
                self.scale3d = scale0
            else:
                return {"status": "calibrating", "frame": emit_idx}

        aligned = np.einsum("ij,pj->pi", self.R_body.T, hip_centered_3d / self.scale3d)
        if self.emit_offset is None:
            self.emit_offset = emit_idx  # 이 세션에서 처음 aligned 프레임이 나온 시점의 emit_idx
            self.current_phase_start = emit_idx
        self.aligned_seq.append(aligned)
        t = emit_idx  # 이후 모든 phase/rep 인덱스는 "원본 스트림(emit) 인덱스" 기준으로 통일

        emitted_2d = np.asarray(self.raw2d_buffer[emit_idx])
        angle_snapshot = self._pose_angle_snapshot(emitted_2d, aligned)
        semantic_cfg = self.weights_cfg.get("2d_semantic_validation", {})
        ema_alpha = float(semantic_cfg.get("ema_alpha", 0.35))
        raw_knee_2d = angle_snapshot["knee_angle_2d"]
        raw_hip_2d = angle_snapshot["hip_angle_2d"]
        self.smoothed_knee_angle_2d = (
            raw_knee_2d if self.smoothed_knee_angle_2d is None
            else ema_alpha * raw_knee_2d + (1.0 - ema_alpha) * self.smoothed_knee_angle_2d
        )
        self.smoothed_hip_angle_2d = (
            raw_hip_2d if self.smoothed_hip_angle_2d is None
            else ema_alpha * raw_hip_2d + (1.0 - ema_alpha) * self.smoothed_hip_angle_2d
        )
        self.knee_angle_2d_hist.append(self.smoothed_knee_angle_2d)
        self.hip_angle_2d_hist.append(self.smoothed_hip_angle_2d)
        self.knee_angle_3d_hist.append(angle_snapshot["knee_angle_3d"])
        self.hip_angle_3d_hist.append(angle_snapshot["hip_angle_3d"])
        pose_angle = (angle_snapshot["hip_angle_2d"] + angle_snapshot["knee_angle_2d"]) / 2.0
        self.pose_angle_hist.append(pose_angle)

        ankle_vert = (aligned[_L_ANKLE, _VERTICAL_AXIS] + aligned[_R_ANKLE, _VERTICAL_AXIS]) / 2.0
        raw_pelvis_height = -ankle_vert
        self.pelvis_height_raw_hist.append(raw_pelvis_height)
        pelvis_height = float(np.median(self.pelvis_height_raw_hist[-3:]))
        self.pelvis_height_hist.append(pelvis_height)
        if self.baseline_height is None and self.state == "prep":
            self.last_baseline_candidate = self.baseline_calibrator.observe(
                t, angle_snapshot, pelvis_height, angle_snapshot
            )
            if self.baseline_calibrator.ready:
                baseline = self.baseline_calibrator.result()
                self.baseline_height = baseline["height"]
                self.baseline_knee_angle_2d = baseline["knee_2d"]
                self.baseline_hip_angle_2d = baseline["hip_2d"]
                self.baseline_knee_angle_3d = baseline["knee_3d"]
                self.baseline_hip_angle_3d = baseline["hip_3d"]
                self.baseline_pose_angle = (baseline["knee_2d"] + baseline["hip_2d"]) / 2.0
                self.baseline_calibration_result = baseline
                print("\n========== STANDING BASELINE ==========", flush=True)
                print(f"accepted frames: {baseline['frames']}", flush=True)
                print(f"rejected frames: {baseline['rejected_total']}", flush=True)
                print(f"hip 2D: {baseline['hip_2d']:.2f} deg", flush=True)
                print(f"knee 2D: {baseline['knee_2d']:.2f} deg", flush=True)
                print("=======================================\n", flush=True)

        velocity = self._causal_velocity()
        flexion_delta = (
            self.baseline_pose_angle - pose_angle
            if self.baseline_pose_angle is not None
            else 0.0
        )
        knee_excursion_2d = (
            self.baseline_knee_angle_2d - self.smoothed_knee_angle_2d
            if self.baseline_knee_angle_2d is not None else 0.0
        )
        hip_excursion_2d = (
            self.baseline_hip_angle_2d - self.smoothed_hip_angle_2d
            if self.baseline_hip_angle_2d is not None else 0.0
        )
        if self.state != "prep":
            self.rep_min_pelvis_height = (
                pelvis_height
                if self.rep_min_pelvis_height is None
                else min(self.rep_min_pelvis_height, pelvis_height)
            )
        self._last_finalize_ms = 0.0
        self._last_semantic_ms = 0.0
        self._last_final_dtw_ms = 0.0
        detector_started = time.perf_counter()
        previous_state = self.state
        # A REP cannot ever finish without a standing baseline because rep_end
        # requires a return to 85% of that baseline.  Keep collecting stable
        # prep frames instead of allowing an unfinishable REP to start.
        if self.baseline_height is None or self.baseline_knee_angle_2d is None:
            self.state = "prep"
            self.debounce_ctr = 0
            event = None
        else:
            if self.rep_detector_mode == "pelvis_fallback":
                event = self._update_phase_state(t, velocity)
                if event == "rep_end":
                    self.rep_detector_mode = "adaptive_2d"
                    self.rep_detector_2d.reset()
            else:
                event = self._update_phase_state_2d(
                    t, velocity, knee_excursion_2d, hip_excursion_2d
                )
                # Regression fallback: if the live 2D domain never crosses the
                # reference-derived start excursion, retain the previously proven
                # pelvis detector as a secondary route.  Once either route starts,
                # one detector owns the REP through rep_end.
                if event is None and self.state == "prep":
                    fallback_event = self._update_phase_state(t, velocity)
                    if fallback_event == "rep_start":
                        self.rep_detector_mode = "pelvis_fallback"
                        self.rep_detector_2d.reset()
                        event = fallback_event
                elif event == "rep_start":
                    self.debounce_ctr = 0
        if event == "rep_start":
            self.rep_min_pelvis_height = pelvis_height

        detector_finished = time.perf_counter()
        frame_timing["rep_detector_ms"] = max(
            0.0,
            (detector_finished - detector_started) * 1000.0
            - float(getattr(self, "_last_finalize_ms", 0.0)),
        )
        partial_started = time.perf_counter()
        partial = (
            self._throttled_partial_online_distance(t, event)
            if self.enable_partial_dtw else None
        )
        frame_timing["partial_dtw_ms"] = (time.perf_counter() - partial_started) * 1000.0
        frame_timing["semantic_ms"] = float(getattr(self, "_last_semantic_ms", 0.0))
        frame_timing["final_dtw_ms"] = float(getattr(self, "_last_final_dtw_ms", 0.0))
        state_debug = self._state_transition_debug(previous_state, velocity)
        if self.rep_detector_mode == "adaptive_2d" and self.rep_detector_2d.last_debug:
            adaptive = dict(self.rep_detector_2d.last_debug)
            evaluated = adaptive["evaluated_state"]
            next_state = {
                "prep": "descend", "descend": "bottom", "bottom": "ascend", "ascend": "completed"
            }.get(evaluated, "-")
            reason = {
                "prep": "2D progress start 또는 pelvis 보조 start 조건",
                "descend": "peak progress에서 reversal 기준 이상 감소",
                "bottom": "progress가 peak보다 감소",
                "ascend": "2D progress 복귀 또는 pelvis 보조 복귀",
            }.get(evaluated, "adaptive 2D condition")
            required = {
                "prep": f"progress >= {self.rep_detector_2d.start_fraction:.3f} OR assisted start",
                "descend": f"drop_from_peak >= {self.rep_detector_2d.reversal_fraction:.3f}",
                "bottom": "progress < peak_progress",
                "ascend": (
                    f"progress <= {self.rep_detector_2d.recovery_fraction:.3f} OR "
                    f"(near_baseline AND progress <= {self.rep_detector_2d.valid_rep_fraction:.3f})"
                ),
            }.get(evaluated, "-")
            state_debug.update(adaptive)
            state_debug.update({
                "detector_mode": "adaptive_2d",
                "next_state": next_state,
                "reason": reason,
                "actual": float(adaptive["combined_progress"]),
                "required": required,
                "condition_passed": bool(adaptive["condition_passed"]),
                "debounce_count": int(adaptive["counter"]),
                "debounce_required": int(adaptive["required_counter"]),
            })
        else:
            state_debug["detector_mode"] = "pelvis_fallback"
        state_debug.update(
            {
                "previous_state": previous_state,
                "current_state": self.state,
                "event": event,
                "frame": t,
                "calibration_ready": self.baseline_height is not None,
                "knee_angle_2d": float(self.smoothed_knee_angle_2d),
                "hip_angle_2d": float(self.smoothed_hip_angle_2d),
                "baseline_knee_angle_2d": self.baseline_knee_angle_2d,
                "baseline_hip_angle_2d": self.baseline_hip_angle_2d,
                "knee_excursion_2d": float(knee_excursion_2d),
                "hip_excursion_2d": float(hip_excursion_2d),
                "framing_ok": True,
                "baseline_candidate": self.last_baseline_candidate,
                "baseline_calibration_result": self.baseline_calibration_result,
            }
        )
        state_debug.update(dict(diagnostic_context or {}))
        state_debug["rep_phase_internal"] = self.state
        state_debug["rep_phase_display"] = rep_phase_label(self.state)
        if self.diagnostic is not None:
            try:
                self.diagnostic.record_detector(state_debug)
            except Exception as error:
                print(f"[REP DIAGNOSTIC WARNING] {type(error).__name__}: {error}", flush=True)
        if event == "rep_start" or self.state != previous_state:
            print("\n========== ANALYSIS LOG TRACE ==========", flush=True)
            print(f"rep_start: {event == 'rep_start'}", flush=True)
            print(f"bottom: {self.state == 'bottom' and previous_state != 'bottom'}", flush=True)
            print(f"ascend: {self.state == 'ascend' and previous_state != 'ascend'}", flush=True)
            print(f"rep_end: {event == 'rep_end'}", flush=True)
            print(f"baseline ready: {self.baseline_height is not None}", flush=True)
            print("========================================\n", flush=True)
        depth_ready = self.state in ("bottom", "ascend") or event == "rep_end"
        hip_down_error = None
        if depth_ready and self.normal_depth_threshold is not None and self.rep_min_pelvis_height is not None:
            hip_down_error = self.rep_min_pelvis_height > self.normal_depth_threshold
        depth_debug = {
            "pelvis_height": pelvis_height,
            "raw_pelvis_height": raw_pelvis_height,
            "rep_min_pelvis_height": self.rep_min_pelvis_height,
            "baseline_height": self.baseline_height,
            "squat_depth": (
                self.baseline_height - self.rep_min_pelvis_height
                if self.baseline_height is not None and self.rep_min_pelvis_height is not None
                else None
            ),
            "normal_depth_threshold": self.normal_depth_threshold,
            # Compatibility aliases: these remain the averaged 3D lifting angles.
            "hip_angle": angle_snapshot["hip_angle_3d"],
            "knee_angle": angle_snapshot["knee_angle_3d"],
            "hip_down_error": hip_down_error,
            "emit_frame": emit_idx,
            "pose_angle_2d": pose_angle,
            "pose_flexion_delta_2d": flexion_delta,
            "knee_excursion_2d": knee_excursion_2d,
            "hip_excursion_2d": hip_excursion_2d,
        }
        depth_debug.update(angle_snapshot)

        realtime_posture_match = {
            "match_valid": False,
            "match_percent": None,
            "phase": self.state,
            "invalid_reason": "matcher_unavailable",
            "calculation_ms": 0.0,
        }
        if self.realtime_posture_matcher is not None:
            realtime_posture_match = self.realtime_posture_matcher.update(
                emitted_2d,
                self.visibility_buffer[emit_idx],
                phase=self.state,
                baseline_ready=self.baseline_height is not None,
            )
            frame_timing["realtime_match_ms"] = float(
                realtime_posture_match.get("calculation_ms", 0.0)
            )

        completed_rep = self._take_completed_analysis()
        return {
            "status": "ok",
            "emit_frame": emit_idx,
            "session_frame": t,
            "phase": self.state,
            "pelvis_height": pelvis_height,
            "velocity": velocity,
            "event": event,
            "partial_distance": partial,
            "completed_rep": completed_rep,
            "depth_debug": depth_debug,
            "state_debug": state_debug,
            "timing": dict(frame_timing),
            "score_ready_rep": None,
            "realtime_posture_match": realtime_posture_match,
        }

    # ------------------------------------------------------------------
    def _causal_velocity(self, win: int = 5) -> float:
        h = self.pelvis_height_hist
        if len(h) < 2:
            return 0.0
        a = h[-min(win, len(h)) :]
        return float(a[-1] - a[0]) / max(1, len(a) - 1)

    @staticmethod
    def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        v1, v2 = a - b, c - b
        cosine = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    @classmethod
    def _pose_angle_snapshot(cls, coords2d: np.ndarray, coords3d: np.ndarray) -> dict:
        """Return same-frame 2D/3D leg angles and diagnostic joint coordinates."""
        values = {}
        for dimension, coords in (("2d", coords2d), ("3d", coords3d)):
            left_hip = cls._joint_angle_deg(coords[_NECK], coords[_L_HIP], coords[_L_KNEE])
            right_hip = cls._joint_angle_deg(coords[_NECK], coords[_R_HIP], coords[_R_KNEE])
            left_knee = cls._joint_angle_deg(coords[_L_HIP], coords[_L_KNEE], coords[_L_ANKLE])
            right_knee = cls._joint_angle_deg(coords[_R_HIP], coords[_R_KNEE], coords[_R_ANKLE])
            values.update(
                {
                    f"left_hip_angle_{dimension}": left_hip,
                    f"right_hip_angle_{dimension}": right_hip,
                    f"hip_angle_{dimension}": (left_hip + right_hip) / 2.0,
                    f"left_knee_angle_{dimension}": left_knee,
                    f"right_knee_angle_{dimension}": right_knee,
                    f"knee_angle_{dimension}": (left_knee + right_knee) / 2.0,
                }
            )
        values["landmarks_2d"] = {
            "left_hip": coords2d[_L_HIP].astype(float).tolist(),
            "right_hip": coords2d[_R_HIP].astype(float).tolist(),
            "left_knee": coords2d[_L_KNEE].astype(float).tolist(),
            "right_knee": coords2d[_R_KNEE].astype(float).tolist(),
            "left_ankle": coords2d[_L_ANKLE].astype(float).tolist(),
            "right_ankle": coords2d[_R_ANKLE].astype(float).tolist(),
        }
        return values

    def _depth_aware_class(
        self, distances: dict[str, float], *, depth_ready: bool
    ) -> str | None:
        """Validate only the shallow-depth class against normalized depth evidence."""
        predicted = min(distances, key=distances.get)
        if predicted != "엉덩이하방오류":
            return predicted
        if not depth_ready:
            return None
        if (
            self.normal_depth_threshold is not None
            and self.rep_min_pelvis_height is not None
            and self.rep_min_pelvis_height <= self.normal_depth_threshold
        ):
            alternatives = {c: distance for c, distance in distances.items() if c != "엉덩이하방오류"}
            return min(alternatives, key=alternatives.get) if alternatives else None
        return predicted

    def _state_transition_debug(self, evaluated_state: str, velocity: float) -> dict:
        near_baseline = (
            self.baseline_height is not None
            and bool(self.pelvis_height_hist)
            and self.pelvis_height_hist[-1] >= 0.85 * self.baseline_height
        )
        if evaluated_state == "prep":
            condition = velocity < -self.vel_eps
            next_state = "descend"
            reason = "pelvis velocity가 하강 기준보다 작아야 함"
            actual = velocity
            required = f"< {-self.vel_eps:.6f}"
        elif evaluated_state == "descend":
            condition = abs(velocity) <= self.vel_eps
            next_state = "bottom"
            reason = "최저점에서 pelvis velocity 절댓값이 정지 기준 이하여야 함"
            actual = abs(velocity)
            required = f"<= {self.vel_eps:.6f}"
        elif evaluated_state == "bottom":
            condition = velocity > self.vel_eps
            next_state = "ascend"
            reason = "pelvis velocity가 상승 기준보다 커야 함"
            actual = velocity
            required = f"> {self.vel_eps:.6f}"
        else:
            stationary = abs(velocity) <= self.vel_eps
            condition = stationary and near_baseline
            next_state = "completed"
            reason = "pelvis가 시작 높이의 85% 이상으로 복귀하고 정지해야 함"
            actual = abs(velocity)
            required = f"|velocity| <= {self.vel_eps:.6f} AND pelvis >= 0.85*baseline"
        return {
            "evaluated_state": evaluated_state,
            "next_state": next_state,
            "reason": reason,
            "actual": float(actual),
            "required": required,
            "condition_passed": bool(condition),
            "debounce_count": self.debounce_ctr,
            "debounce_required": self.debounce_n,
            "pelvis_height": float(self.pelvis_height_hist[-1]),
            "pelvis_velocity": float(velocity),
            "baseline_height": self.baseline_height,
            "return_height_threshold": (
                0.85 * self.baseline_height if self.baseline_height is not None else None
            ),
            "near_baseline": bool(near_baseline),
            "pose_flexion_delta_2d": float(
                (self.baseline_pose_angle or 0.0) - self.pose_angle_hist[-1]
                if self.pose_angle_hist else 0.0
            ),
        }

    def _update_phase_state_2d(
        self, t: int, velocity: float, knee_excursion: float, hip_excursion: float
    ) -> str | None:
        """Drive REP phases from adaptive 2D flexion; 3D pelvis is supporting only."""
        near_baseline = (
            self.baseline_height is not None
            and self.pelvis_height_hist[-1] >= 0.85 * self.baseline_height
        )
        event = self.rep_detector_2d.update(
            t,
            knee_excursion,
            hip_excursion,
            pelvis_velocity=velocity,
            pelvis_velocity_eps=self.vel_eps,
            pelvis_near_baseline=near_baseline,
        )
        if event == "rep_start":
            self.state = "descend"
            self.rep_start_idx = self.rep_detector_2d.rep_start_frame
            self.phase_boundaries_running = {"준비": [self.current_phase_start, self.rep_start_idx]}
            self.current_phase_start = self.rep_start_idx
        elif event == "bottom":
            boundary = max(self.current_phase_start, t - 1)
            self.phase_boundaries_running["하강"] = [self.current_phase_start, boundary]
            self.current_phase_start = boundary
            self.state = "bottom"
        elif event == "ascend":
            self.phase_boundaries_running["최저점"] = [self.current_phase_start, t]
            self.current_phase_start = t
            self.state = "ascend"
        elif event == "abort":
            self.state = "prep"
            self.rep_start_idx = None
            self.phase_boundaries_running = {}
            self.current_phase_start = t + 1
        elif event == "rep_end":
            self.phase_boundaries_running["상승"] = [self.current_phase_start, max(self.current_phase_start, t - 2)]
            self.phase_boundaries_running["종료"] = [max(self.current_phase_start, t - 2), t + 1]
            self._finalize_rep(t)
            self.state = "prep"
            self.current_phase_start = t + 1
        return event

    def _update_phase_state(self, t: int, velocity: float) -> str | None:
        """Legacy pelvis-only detector kept for offline compatibility tests."""
        event = None
        if self.state == "prep":
            if velocity < -self.vel_eps:
                self.debounce_ctr += 1
            else:
                self.debounce_ctr = 0
            if self.debounce_ctr >= self.debounce_n:
                self.state = "descend"
                self.rep_start_idx = max(0, t - self.debounce_n)
                self.phase_boundaries_running = {"준비": [self.current_phase_start, self.rep_start_idx]}
                self.current_phase_start = self.rep_start_idx
                self.debounce_ctr = 0
                event = "rep_start"
        elif self.state == "descend":
            if abs(velocity) <= self.vel_eps:
                self.debounce_ctr += 1
            else:
                self.debounce_ctr = 0
            if self.debounce_ctr >= self.debounce_n:
                self.phase_boundaries_running["하강"] = [self.current_phase_start, t - self.debounce_n]
                self.current_phase_start = t - self.debounce_n
                self.state = "bottom"
                self.debounce_ctr = 0
        elif self.state == "bottom":
            if velocity > self.vel_eps:
                self.debounce_ctr += 1
            else:
                self.debounce_ctr = 0
            if self.debounce_ctr >= self.debounce_n:
                self.phase_boundaries_running["최저점"] = [self.current_phase_start, t - self.debounce_n]
                self.current_phase_start = t - self.debounce_n
                self.state = "ascend"
                self.debounce_ctr = 0
        elif self.state == "ascend":
            near_baseline = self.baseline_height is not None and self.pelvis_height_hist[-1] >= 0.85 * self.baseline_height
            if abs(velocity) <= self.vel_eps and near_baseline:
                self.debounce_ctr += 1
            else:
                self.debounce_ctr = 0
            if self.debounce_ctr >= self.debounce_n:
                self.phase_boundaries_running["상승"] = [self.current_phase_start, t - self.debounce_n]
                self.phase_boundaries_running["종료"] = [t - self.debounce_n, t + 1]
                event = "rep_end"
                self._finalize_rep(t)
                self.state = "prep"
                self.current_phase_start = t + 1
                self.debounce_ctr = 0
        return event

    def _arr_idx(self, t: int) -> int:
        """emit_idx(원본 스트림 인덱스) -> self.aligned_seq 리스트 인덱스 변환."""
        return max(0, t - self.emit_offset)

    def _finalize_rep(self, end_t: int) -> None:
        """Snapshot a completed REP and return immediately to the camera loop."""
        submit_started = time.perf_counter()
        rep_index = self.submitted_rep_count
        self.submitted_rep_count += 1
        snapshot = copy.copy(self)
        snapshot.raw2d_buffer = [np.asarray(frame).copy() for frame in self.raw2d_buffer]
        snapshot.visibility_buffer = [dict(row) for row in self.visibility_buffer]
        snapshot.aligned_seq = [np.asarray(frame).copy() for frame in self.aligned_seq]
        snapshot.knee_angle_2d_hist = list(self.knee_angle_2d_hist)
        snapshot.hip_angle_2d_hist = list(self.hip_angle_2d_hist)
        snapshot.knee_angle_3d_hist = list(self.knee_angle_3d_hist)
        snapshot.hip_angle_3d_hist = list(self.hip_angle_3d_hist)
        snapshot.phase_boundaries_running = copy.deepcopy(self.phase_boundaries_running)
        snapshot.baseline_calibration_result = copy.deepcopy(self.baseline_calibration_result)
        snapshot.completed_reps = [None] * rep_index
        snapshot.scored_results = []
        snapshot.score_futures = []
        snapshot.diagnostic_futures = []
        submitted_at = time.perf_counter()
        pending = sum(not future.done() for future in self.final_analysis_futures)
        self.final_analysis_max_queue_depth = max(self.final_analysis_max_queue_depth, pending + 1)
        future = self.final_analysis_executor.submit(
            self._run_final_analysis, snapshot, end_t, rep_index, submitted_at
        )
        self.final_analysis_futures.append(future)
        self._last_finalize_ms = (time.perf_counter() - submit_started) * 1000.0
        print(
            f"[FINAL ANALYSIS SUBMIT] rep={rep_index + 1} queue_depth={pending + 1} "
            f"snapshot_ms={self._last_finalize_ms:.3f}", flush=True,
        )

    def _run_final_analysis(self, snapshot, end_t: int, rep_index: int, submitted_at: float) -> None:
        started_at = time.perf_counter()
        try:
            snapshot._finalize_rep_sync(end_t)
            for future in snapshot.score_futures:
                future.result()
            result = snapshot.completed_reps[-1]
            finished_at = time.perf_counter()
            timing = {
                "rep_index": rep_index,
                "submitted_at": submitted_at,
                "started_at": started_at,
                "finished_at": finished_at,
                "queue_wait_ms": (started_at - submitted_at) * 1000.0,
                "analysis_ms": (finished_at - started_at) * 1000.0,
                "final_dtw_ms": float(getattr(snapshot, "_last_final_dtw_ms", 0.0)),
            }
            self.final_analysis_timings.append(timing)
            self.final_analysis_results.put((rep_index, result))
            print(
                f"[FINAL ANALYSIS READY] rep={rep_index + 1} "
                f"queue_wait_ms={timing['queue_wait_ms']:.3f} "
                f"analysis_ms={timing['analysis_ms']:.3f}", flush=True,
            )
        except Exception as error:
            print(
                f"[FINAL ANALYSIS ERROR] rep={rep_index + 1} "
                f"{type(error).__name__}: {error}", flush=True,
            )

    def _take_completed_analysis(self):
        try:
            rep_index, result = self.final_analysis_results.get_nowait()
        except queue.Empty:
            return None
        if rep_index != len(self.completed_reps):
            raise RuntimeError(
                f"Final analysis result order mismatch: expected {len(self.completed_reps)}, got {rep_index}"
            )
        self.completed_reps.append(result)
        return result

    def _finalize_rep_sync(self, end_t: int) -> None:
        finalize_started = time.perf_counter()
        print("\n========== ANALYSIS LOG TRACE ==========", flush=True)
        print("_finalize_rep called: True", flush=True)
        print(f"rep_end frame: {end_t}", flush=True)
        print("========================================\n", flush=True)
        start = self.rep_start_idx if self.rep_start_idx is not None else 0
        rep_coords = np.stack(self.aligned_seq[self._arr_idx(start) : self._arr_idx(end_t) + 1])
        feat = extract_all_features(rep_coords)
        bounds = {p: [max(0, s - start), max(0, e - start)] for p, (s, e) in self.phase_boundaries_running.items()}
        for p in PHASES:
            bounds.setdefault(p, [0, 0])

        final_dtw_started = time.perf_counter()
        per_class = {}
        for cls, medoids in self.db_operational.items():
            w = resolve_weights(self.weights_cfg, self.weight_profile, class_label=cls)
            per_class[cls] = multi_reference_distance(feat, bounds, medoids, w, self.weights_cfg, top_k=2)
        class_distances = {c: result["min_distance"] for c, result in per_class.items()}
        self._last_final_dtw_ms = (time.perf_counter() - final_dtw_started) * 1000.0
        raw_pred = min(class_distances, key=class_distances.get)
        score = None
        if self.score_calib is not None:
            from .scoring import distance_to_score

            score = distance_to_score(per_class["정상"]["min_distance"], self.score_calib)
        pelvis_values = feat["pelvis_trajectory"][:, 0]
        hip_angles = feat["hip_flexion_angle"].mean(axis=1) * 180.0
        knee_angles = feat["knee_flexion_angle"].mean(axis=1) * 180.0
        rep_slice = slice(self._arr_idx(start), self._arr_idx(end_t) + 1)
        knee_2d = np.asarray(self.knee_angle_2d_hist[rep_slice], dtype=float)
        hip_2d = np.asarray(self.hip_angle_2d_hist[rep_slice], dtype=float)
        knee_excursion_2d = float(self.baseline_knee_angle_2d - np.nanmin(knee_2d))
        hip_excursion_2d = float(self.baseline_hip_angle_2d - np.nanmin(hip_2d))
        knee_excursion_3d = float(self.baseline_knee_angle_3d - np.nanmin(knee_angles))
        hip_excursion_3d = float(self.baseline_hip_angle_3d - np.nanmin(hip_angles))
        semantic_cfg = self.weights_cfg.get("2d_semantic_validation", {})
        knee_required = float(semantic_cfg["knee_excursion_min_deg"])
        hip_required = float(semantic_cfg["hip_excursion_min_deg"])
        depth_2d_pass = knee_excursion_2d >= knee_required and hip_excursion_2d >= hip_required
        knee_consistency_delta = knee_excursion_3d - knee_excursion_2d
        hip_consistency_delta = hip_excursion_3d - hip_excursion_2d
        consistency_pass = (
            float(semantic_cfg["knee_3d_minus_2d_excursion_min_deg"])
            <= knee_consistency_delta
            <= float(semantic_cfg["knee_3d_minus_2d_excursion_max_deg"])
            and float(semantic_cfg["hip_3d_minus_2d_excursion_min_deg"])
            <= hip_consistency_delta
            <= float(semantic_cfg["hip_3d_minus_2d_excursion_max_deg"])
        )
        knee_consistency_pass = (
            float(semantic_cfg["knee_3d_minus_2d_excursion_min_deg"])
            <= knee_consistency_delta
            <= float(semantic_cfg["knee_3d_minus_2d_excursion_max_deg"])
        )
        hip_consistency_pass = (
            float(semantic_cfg["hip_3d_minus_2d_excursion_min_deg"])
            <= hip_consistency_delta
            <= float(semantic_cfg["hip_3d_minus_2d_excursion_max_deg"])
        )
        progress = (
            np.maximum(0.0, self.baseline_knee_angle_2d - knee_2d) / knee_required
            + np.maximum(0.0, self.baseline_hip_angle_2d - hip_2d) / hip_required
        ) / 2.0
        semantic_bottom_idx = int(np.nanargmax(progress))
        pelvis_bottom_idx = int(np.nanargmin(pelvis_values))
        recomputed_min = float(pelvis_values[pelvis_bottom_idx])
        bottom_stream_frame = start + semantic_bottom_idx
        bottom_snapshot = self._pose_angle_snapshot(
            np.asarray(self.raw2d_buffer[bottom_stream_frame]), rep_coords[semantic_bottom_idx]
        )
        bottom_snapshot.update(
            {"pelvis_height": float(pelvis_values[semantic_bottom_idx]), "emit_frame": bottom_stream_frame}
        )
        root_centered_2d = np.asarray(self.raw2d_buffer[bottom_stream_frame], dtype=float)
        root_centered_2d = root_centered_2d - root_centered_2d[_IDX["Hip"]]
        bottom_snapshot["root_centered_2d"] = {
            name: root_centered_2d[_IDX[name]].tolist()
            for name in ("LHip", "RHip", "LKnee", "RKnee", "LAnkle", "RAnkle")
        }
        bottom_snapshot["visibility"] = (
            self.visibility_buffer[bottom_stream_frame]
            if bottom_stream_frame < len(self.visibility_buffer) else {}
        )

        # The lifting output at stream frame t is produced from [t-4..t+4]
        # but represents center t.  Compare nearby raw 2D angles diagnostically;
        # this does not affect classification.
        offset_alignment = []
        bottom_3d = rep_coords[semantic_bottom_idx]
        for offset in range(-_HALF, _HALF + 1):
            raw_index = bottom_stream_frame + offset
            if not 0 <= raw_index < len(self.raw2d_buffer):
                continue
            snapshot = self._pose_angle_snapshot(
                np.asarray(self.raw2d_buffer[raw_index]), bottom_3d
            )
            angle_error = abs(snapshot["knee_angle_3d"] - snapshot["knee_angle_2d"]) + abs(
                snapshot["hip_angle_3d"] - snapshot["hip_angle_2d"]
            )
            offset_alignment.append(
                {
                    "offset": offset,
                    "knee_2d": snapshot["knee_angle_2d"],
                    "hip_2d": snapshot["hip_angle_2d"],
                    "angle_error": angle_error,
                }
            )
        best_offset = min(offset_alignment, key=lambda row: row["angle_error"])["offset"]
        semantic_started = time.perf_counter()
        sorted_classes = sorted(class_distances, key=class_distances.get)
        margin = (
            float(class_distances[sorted_classes[1]] - class_distances[sorted_classes[0]])
            if len(sorted_classes) > 1 else 0.0
        )
        relative_margin = margin / max(float(class_distances[sorted_classes[0]]), 1e-8)
        pred = validate_production_candidate(
            raw_pred, depth_2d_pass=depth_2d_pass, consistency_pass=consistency_pass
        )
        pre_heel_pred = pred
        from .heel_semantic import heel_motion_evidence, validate_heel_candidate
        heel_cfg = self.weights_cfg.get("heel_semantic_validation", {})
        heel_evidence = heel_motion_evidence(
            np.asarray(self.raw2d_buffer[start:end_t + 1]),
            no_max=float(heel_cfg.get("no_evidence_max", 0.08158551080553868)),
            yes_min=float(heel_cfg.get("positive_evidence_min", 0.10469323648288038)),
        )
        pred, heel_reason_code = validate_heel_candidate(
            raw_pred,
            pred,
            heel_evidence,
            relative_margin,
        )
        if raw_pred == "엉덩이하방오류" and depth_2d_pass:
            reason_code = "UNKNOWN_DEPTH_CONFLICT"
            reason = "DTW 하방오류 후보와 달리 2D 굽힘은 정상 reference 최소 excursion을 통과함"
        elif raw_pred == "정상" and not depth_2d_pass:
            reason_code = "UNKNOWN_DEPTH_AMBIGUOUS"
            reason = "DTW는 정상이지만 2D 굽힘이 정상 reference 최소 excursion을 통과하지 못함"
        else:
            reason_code = (
                "VALIDATED" if consistency_pass
                else "VALIDATED_2D_3D_MISMATCH_DIAGNOSTIC"
            )
            reason = (
                "DTW 후보가 활성 semantic guard를 통과함; 2D/3D 불일치는 진단용"
                if not consistency_pass
                else "DTW 후보가 활성 semantic guard를 통과함"
            )
        if heel_reason_code is not None:
            reason_code = heel_reason_code
            reason = "DTW 발뒤꿈치오류 후보와 MediaPipe 2D heel 움직임 근거가 일치하지 않음"
        self._last_semantic_ms = (time.perf_counter() - semantic_started) * 1000.0
        top_feat = sorted(
            per_class[raw_pred]["best_detail"]["per_feature_contrib"].items(), key=lambda kv: -kv[1]
        )[:3]
        depth_pass = (
            self.normal_depth_threshold is not None
            and np.isfinite(recomputed_min)
            and recomputed_min <= self.normal_depth_threshold
        )
        def phase_at(index: int) -> str:
            for phase_name, (phase_start, phase_end) in bounds.items():
                if phase_start <= index < phase_end:
                    return phase_name
            return "-"

        running_min = np.minimum.accumulate(pelvis_values)
        debug_trace = [
            {
                "frame": start + index,
                "rep_frame": index,
                "phase": phase_at(index),
                "pelvis_height": float(pelvis_values[index]),
                "rep_min_pelvis_height": float(running_min[index]),
                "hip_angle": float(hip_angles[index]),
                "knee_angle": float(knee_angles[index]),
            }
            for index in range(len(pelvis_values))
        ]
        debug_summary = {
            "standing_pelvis_height": self.baseline_height,
            "tracked_min_pelvis_height": self.rep_min_pelvis_height,
            "recomputed_min_pelvis_height": recomputed_min,
            "bottom_rep_frame": semantic_bottom_idx,
            "pelvis_bottom_rep_frame": pelvis_bottom_idx,
            "bottom_stream_frame": bottom_stream_frame,
            "bottom_hip_angle": float(hip_angles[semantic_bottom_idx]),
            "bottom_knee_angle": float(knee_angles[semantic_bottom_idx]),
            "bottom_pose": bottom_snapshot,
            "normal_depth_threshold": self.normal_depth_threshold,
            "depth_difference": (
                recomputed_min - self.normal_depth_threshold
                if self.normal_depth_threshold is not None
                else None
            ),
            "depth_pass": bool(depth_pass),
            "knee_excursion_2d": knee_excursion_2d,
            "hip_excursion_2d": hip_excursion_2d,
            "knee_excursion_3d": knee_excursion_3d,
            "hip_excursion_3d": hip_excursion_3d,
            "knee_excursion_required_2d": knee_required,
            "hip_excursion_required_2d": hip_required,
            "depth_2d_pass": bool(depth_2d_pass),
            "knee_consistency_delta": knee_consistency_delta,
            "hip_consistency_delta": hip_consistency_delta,
            "consistency_pass": bool(consistency_pass),
            "knee_consistency_pass": bool(knee_consistency_pass),
            "hip_consistency_pass": bool(hip_consistency_pass),
            "knee_consistency_range": [
                float(semantic_cfg["knee_3d_minus_2d_excursion_min_deg"]),
                float(semantic_cfg["knee_3d_minus_2d_excursion_max_deg"]),
            ],
            "hip_consistency_range": [
                float(semantic_cfg["hip_3d_minus_2d_excursion_min_deg"]),
                float(semantic_cfg["hip_3d_minus_2d_excursion_max_deg"]),
            ],
            "standing_knee_angle_2d": self.baseline_knee_angle_2d,
            "bottom_knee_angle_2d": float(np.nanmin(knee_2d)),
            "standing_hip_angle_2d": self.baseline_hip_angle_2d,
            "bottom_hip_angle_2d": float(np.nanmin(hip_2d)),
            "standing_knee_angle_3d": self.baseline_knee_angle_3d,
            "bottom_knee_angle_3d": float(np.nanmin(knee_angles)),
            "standing_hip_angle_3d": self.baseline_hip_angle_3d,
            "bottom_hip_angle_3d": float(np.nanmin(hip_angles)),
            "temporal_alignment_current_offset": 0,
            "temporal_alignment_best_offset": int(best_offset),
            "temporal_alignment_offsets": offset_alignment,
            "input_quality": "PASS" if consistency_pass else "UNRELIABLE_3D",
            "raw_dtw_predicted_class": raw_pred,
            "final_predicted_class": pred,
            "class_distances": class_distances,
            "nearest_margin": margin,
            "nearest_relative_margin": relative_margin,
            "nearest_second_class": sorted_classes[1] if len(sorted_classes) > 1 else None,
            "heel_semantic": heel_evidence,
            "reason": reason,
            "reason_code": reason_code,
            "normal_references": self._reference_depth_report("정상"),
            "hip_down_references": self._reference_depth_report("엉덩이하방오류"),
            "values_finite": bool(
                np.all(np.isfinite(pelvis_values))
                and np.all(np.isfinite(hip_angles))
                and np.all(np.isfinite(knee_angles))
            ),
        }

        result = RepResult(
            rep_index=len(self.completed_reps),
            frame_range=(start, end_t),
            predicted_class=pred,
            raw_distance_by_class={c: per_class[c]["min_distance"] for c in per_class},
            score_vs_normal=score,
            top_contributing_features=top_feat,
            posture_score={"pending": True, "score_valid": False},
            debug_summary=debug_summary,
            debug_trace=debug_trace,
        )
        self.completed_reps.append(result)
        if self.posture_scorer is not None:
            baseline_frames = list((self.baseline_calibration_result or {}).get("frames", []))
            standing2d = np.asarray([self.raw2d_buffer[index] for index in baseline_frames]).copy()
            standing3d = np.asarray([
                self.aligned_seq[self._arr_idx(index)] for index in baseline_frames
                if 0 <= self._arr_idx(index) < len(self.aligned_seq)
            ]).copy()
            future = self.score_executor.submit(
                self._calculate_posture_score,
                result,
                np.asarray(self.raw2d_buffer[start:end_t + 1]).copy(),
                rep_coords.copy(),
                [dict(row) for row in self.visibility_buffer[start:end_t + 1]],
                self.baseline_knee_angle_2d,
                self.baseline_hip_angle_2d,
                raw_pred,
                pred,
                standing2d,
                standing3d,
                semantic_bottom_idx,
                {
                    "raw": raw_pred, "final": pred, "reason_code": reason_code,
                    "depth_2d_pass": bool(depth_2d_pass),
                    "consistency_pass": bool(consistency_pass),
                },
            )
            self.score_futures.append(future)
        production = {
                    "rep": result.rep_index + 1,
                    "raw_3d_dtw": raw_pred,
                    "pre_heel_final": pre_heel_pred,
                    "final": pred,
                    "reason_code": reason_code,
                    "depth_2d_pass": bool(depth_2d_pass),
                    "consistency_pass": bool(consistency_pass),
                    "knee_excursion_2d": knee_excursion_2d,
                    "hip_excursion_2d": hip_excursion_2d,
                    "knee_excursion_3d": knee_excursion_3d,
                    "hip_excursion_3d": hip_excursion_3d,
                    "class_distances": class_distances,
                    "class_details": {
                        class_name: {
                            "best_reference": details["best_medoid"].get(
                                "medoid_id", details["best_medoid"].get("array_key", "unknown")
                            ),
                            "per_feature_contrib": details["best_detail"]["per_feature_contrib"],
                            "per_phase": details["best_detail"]["per_phase"],
                        }
                        for class_name, details in per_class.items()
                    },
                }
        if self.heel_validation_logger is not None:
            self.heel_validation_logger.safe_record(
                result=result,
                production=production,
                raw2d=np.asarray(self.raw2d_buffer[start:end_t + 1]).copy(),
                heel_thresholds=self.weights_cfg.get("heel_semantic_validation", {}),
            )
        if self.enable_heavy_diagnostics:
            snapshot = copy.copy(self)
            snapshot.raw2d_buffer = [np.asarray(frame).copy() for frame in self.raw2d_buffer[: end_t + 1]]
            snapshot.aligned_seq = [np.asarray(frame).copy() for frame in self.aligned_seq[: self._arr_idx(end_t) + 1]]
        bottom_abs = tuple(self.phase_boundaries_running.get("최저점", [start, start + 1]))
        if self.enable_heavy_diagnostics:
            future = self.diagnostic_executor.submit(
                self._run_post_rep_diagnostics,
                snapshot, result, production, start, end_t, bottom_abs,
                dict(bottom_snapshot.get("visibility", {})), dict(semantic_cfg),
            )
            self.diagnostic_futures.append(future)
        self._last_finalize_ms = (time.perf_counter() - finalize_started) * 1000.0
        print("\n========== ANALYSIS LOG TRACE ==========", flush=True)
        print("final result created: True", flush=True)
        print(f"final class: {result.predicted_class}", flush=True)
        print(f"completed rep index: {result.rep_index + 1}", flush=True)
        print("========================================\n", flush=True)

    def _run_post_rep_diagnostics(
        self, snapshot, result, production, start, end_t, bottom_abs,
        visibility, semantic_cfg,
    ) -> None:
        """Run expensive A/B diagnostics after REP finalization, off the camera loop."""
        started = time.perf_counter()
        try:
            diagnostic_start = max(0, start - 5)
            bottom_range = (bottom_abs[0] - diagnostic_start, bottom_abs[1] - diagnostic_start)
            two_d_result = None
            two_d_started = time.perf_counter()
            if self.diagnostic_2d is not None:
                from .adapter_diagnostic import run_hip_width_ab
                normal_geometry = self.diagnostic_2d.report["geometry_distributions"]["정상"]
                hip_target = normal_geometry["prep"]["hip_width/torso_length"]["median"]
                hip_geometry = run_hip_width_ab(
                    snapshot, start, end_t, int(bottom_abs[0] - start),
                    visibility, hip_target, semantic_cfg,
                )
                hip_geometry["normal_reference_distribution"] = normal_geometry
                hip_geometry["production_trigger"] = {
                    "depth_2d_pass": production["depth_2d_pass"],
                    "consistency_pass": production["consistency_pass"],
                    "raw_dtw": production["raw_3d_dtw"],
                    "final": production["final"],
                    "reason_code": production["reason_code"],
                }
                two_d_result = self.diagnostic_2d.diagnose(
                    np.asarray(snapshot.raw2d_buffer[diagnostic_start:end_t + 1]), production,
                    baseline_knee=snapshot.baseline_knee_angle_2d,
                    baseline_hip=snapshot.baseline_hip_angle_2d,
                    bottom_range=bottom_range,
                    hip_geometry=hip_geometry,
                )
                result.debug_summary["2d_only"] = two_d_result
            two_d_compute_ms = (time.perf_counter() - two_d_started) * 1000.0
            if self.heel_shadow_logger is not None:
                try:
                    self.heel_shadow_logger.record(
                        result=result, production=production, two_d=two_d_result,
                        heel_thresholds=self.weights_cfg.get("heel_semantic_validation", {}),
                        two_d_compute_ms=two_d_compute_ms,
                    )
                except Exception as error:
                    print(f"[HEEL-SHADOW WARNING] {type(error).__name__}: {error}", flush=True)

            from .adapter_diagnostic import run_adapter_ab
            detector_bottom = int(bottom_abs[0] - start)
            result.debug_summary["adapter_ab"] = run_adapter_ab(
                snapshot, start, end_t, detector_bottom, visibility
            )
            print(f"[REP DIAGNOSTIC] REP {result.rep_index + 1} complete", flush=True)
        except Exception as error:
            print(f"[REP DIAGNOSTIC WARNING] {type(error).__name__}: {error}", flush=True)
        finally:
            self.diagnostic_timings_ms.append((time.perf_counter() - started) * 1000.0)

    def _calculate_posture_score(
        self, result, raw2d, coords3d, visibility, baseline_knee, baseline_hip,
        production_raw, production_final, standing2d, standing3d, detector_bottom,
        production,
    ) -> None:
        try:
            scored = self.posture_scorer.score(
                raw2d, coords3d, visibility,
                baseline_knee=baseline_knee,
                baseline_hip=baseline_hip,
                production_raw=production_raw,
                production_final=production_final,
                rep_index=result.rep_index + 1,
            )
            result.posture_score = scored.as_dict()
            if self.angle_domain_diagnostic is not None and len(standing2d) and len(standing3d):
                self.angle_domain_diagnostic.record(
                    rep=result.rep_index + 1,
                    raw2d=raw2d,
                    coords3d=coords3d,
                    standing2d_frames=standing2d,
                    standing3d_frames=standing3d,
                    detector_bottom=detector_bottom,
                    scorer_payload=result.posture_score,
                    production=production,
                )
        except Exception as error:
            result.posture_score = {
                "pending": False, "score_valid": False, "measurement_valid": False,
                "invalid_reason": f"{type(error).__name__}: {error}",
            }
        self.scored_results.append(result)

    def close_scoring(self) -> None:
        self.score_executor.shutdown(wait=True, cancel_futures=False)

    def close_final_analysis(self) -> None:
        """Finish ordered final analyses before their dependent score executor closes."""
        self.final_analysis_executor.shutdown(wait=True, cancel_futures=False)
        while self._take_completed_analysis() is not None:
            pass

    def close_diagnostics(self) -> None:
        """Wait only during shutdown; diagnostics never block live camera processing."""
        self.diagnostic_executor.shutdown(wait=True, cancel_futures=False)

    def _reference_depth_report(self, class_label: str) -> list[dict]:
        report = []
        for medoid in self.db_operational.get(class_label, []):
            feat = medoid["feat"]
            pelvis = feat["pelvis_trajectory"][:, 0]
            hip_angles = feat["hip_flexion_angle"].mean(axis=1) * 180.0
            knee_angles = feat["knee_flexion_angle"].mean(axis=1) * 180.0
            bottom_idx = int(np.nanargmin(pelvis))
            meta = medoid.get("meta", {})
            report.append(
                {
                    "id": meta.get("medoid_id", meta.get("array_key", "unknown")),
                    "pelvis_min": float(pelvis[bottom_idx]),
                    "pelvis_max": float(np.nanmax(pelvis)),
                    "standing_pelvis": float(pelvis[0]),
                    "bottom_frame": bottom_idx,
                    "bottom_hip_angle": float(hip_angles[bottom_idx]),
                    "bottom_knee_angle": float(knee_angles[bottom_idx]),
                    "hip_angle_min": float(np.nanmin(hip_angles)),
                    "knee_angle_min": float(np.nanmin(knee_angles)),
                }
            )
        return report

    def _throttled_partial_online_distance(self, t: int, event: str | None) -> dict | None:
        """Reuse partial-DTW between configured checkpoints; final REP DTW is untouched."""
        if self.state == "prep":
            self.partial_dtw_last_frame = None
            self.partial_dtw_last_state = None
            self.partial_dtw_cache = None
            return None

        interval = max(1, int(
            self.weights_cfg.get("temporal_alignment", {}).get(
                "partial_dtw_interval_frames", 6
            )
        ))
        state_changed = self.partial_dtw_last_state != self.state
        due = self.partial_dtw_last_frame is None or t - self.partial_dtw_last_frame >= interval
        if event is not None or state_changed or due:
            self.partial_dtw_cache = self._partial_online_distance(t)
            self.partial_dtw_last_frame = t
            self.partial_dtw_last_state = self.state
        return self.partial_dtw_cache

    def _partial_online_distance(self, t: int) -> dict | None:
        """현재 phase 진입 이후 지금까지의 partial 시퀀스를, 각 클래스 reference의
        동일 phase와 subsequence 방식(끝점 미고정)으로 비교한 distance. 실시간 모니터링용."""
        min_partial_frames = int(
            self.weights_cfg.get("temporal_alignment", {}).get("min_partial_frames", 8)
        )
        if self.state == "prep" or t - self.current_phase_start + 1 < min_partial_frames:
            return None
        phase_name = {"descend": "하강", "bottom": "최저점", "ascend": "상승"}.get(self.state)
        if phase_name is None:
            return None

        partial_coords = np.stack(self.aligned_seq[self._arr_idx(self.current_phase_start) : self._arr_idx(t) + 1])
        partial_feat = extract_all_features(partial_coords)

        from scipy.spatial.distance import cdist

        from .dtw_compare import weighted_frame_cost_matrix

        out = {}
        for cls, medoids in self.db_operational.items():
            w = resolve_weights(self.weights_cfg, self.weight_profile, class_label=cls)
            best = None
            for med in medoids:
                s, e = med["bounds"][phase_name]
                if e <= s:
                    continue
                ref_feat = {k: v[s:e] for k, v in med["feat"].items()}
                cost, _ = weighted_frame_cost_matrix(partial_feat, ref_feat, w, self.weights_cfg)
                if cost.size == 0:
                    continue
                # subsequence DTW: query(부분) 전체는 매칭, reference 끝점은 자유
                d = _subsequence_dtw_last_row_min(cost)
                if best is None or d < best:
                    best = d
            if best is not None:
                out[cls] = best
        if not out:
            return None
        raw_best_class = min(out, key=out.get)
        predicted_class = self._depth_aware_class(out, depth_ready=self.state in ("bottom", "ascend"))
        return {
            "phase": phase_name,
            "distance_by_class": out,
            "raw_best_class": raw_best_class,
            "validated_class": predicted_class,
            # Partial DTW is an unfinished motion. Keep its candidates for live
            # diagnostics, but do not expose a concrete error as a final verdict.
            "predicted_class": None,
        }


def _subsequence_dtw_last_row_min(cost: np.ndarray) -> float:
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    path_len = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[0, :] = 0.0  # reference 시작점 자유 (subsequence)
    for i in range(1, n + 1):
        row = cost[i - 1]
        for j in range(1, m + 1):
            predecessors = (d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])
            best_idx = int(np.argmin(predecessors))
            prev_i, prev_j = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[best_idx]
            d[i, j] = row[j - 1] + predecessors[best_idx]
            path_len[i, j] = path_len[prev_i, prev_j] + 1
    end_j = int(np.argmin(d[n, 1:])) + 1
    return float(d[n, end_j]) / max(1, int(path_len[n, end_j]))
