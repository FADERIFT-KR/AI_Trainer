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

from dataclasses import dataclass, field

import numpy as np
import torch

from .common_skeleton import COMMON_JOINT_NAMES
from .dtw_compare import (
    DTW_ALGORITHMS,
    PHASES,
    derivative_features,
    multi_reference_distance,
    resolve_weights,
)
from .features import _angle_deg, extract_all_features
from .heel_contact import evaluate_heel_contact_2d
from .lifting_dataset import WINDOW_T
from .normalization import body_axes, hip_center_3d, leg_length_scale
from .reference_levels import DIFFICULTY_LEVELS, REFERENCE_CLASSES
from .reference_matching import decide_reference_match

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_L_HIP, _R_HIP = _IDX["LHip"], _IDX["RHip"]
_L_KNEE, _R_KNEE = _IDX["LKnee"], _IDX["RKnee"]
_L_ANKLE, _R_ANKLE = _IDX["LAnkle"], _IDX["RAnkle"]
_L_SHOULDER, _R_SHOULDER = _IDX["LShoulder"], _IDX["RShoulder"]
_L_HEEL, _R_HEEL = _IDX["LHeel"], _IDX["RHeel"]
_L_BIGTOE, _R_BIGTOE = _IDX["LBigToe"], _IDX["RBigToe"]
_HIP, _NECK = _IDX["Hip"], _IDX["Neck"]
_VERTICAL_AXIS = 1
_HALF = WINDOW_T // 2  # =4


@dataclass
class RepResult:
    rep_index: int
    frame_range: tuple[int, int]
    predicted_class: str
    matched_level: str
    match_rate: float | None
    normal_match_by_level: dict[str, float | None]
    normal_distance_by_level: dict[str, float]
    raw_distance_by_level: dict[str, dict[str, float]]
    raw_distance_by_class: dict[str, float]
    score_vs_normal: float | None
    top_contributing_features: list[tuple[str, float]]
    decision_status: str
    rejection_reason: str | None
    error_relative_margin: float | None
    error_z_margin: float | None
    normal_probability: float | None
    heel_contact_2d: bool | None
    heel_lift_max_delta: tuple[float, float] | None


@dataclass
class OnlineSquatSession:
    model: torch.nn.Module
    device: torch.device
    # level -> class -> [{feat,bounds,meta}]
    db_operational: dict[str, dict[str, list[dict]]]
    weights_cfg: dict
    weight_profile: str | None = None  # None -> weights_cfg["default_profile"]
    calib_frames: int = 8
    # 모든 phase 전환에 동일하게 적용하는 초기 기준
    vel_eps: float = 0.004
    debounce_n: int = 0
    score_calib: dict | None = None
    algorithm: str = "dtw"
    # Frame-wise 2D EMA is intentionally short: it suppresses landmark jitter
    # before lifting while retaining the causal character of the stream.
    ema_alpha: float = 0.45
    heel_lift_threshold: float = 0.10  # initial torso-length ratio
    butt_wink_bottom_delta_deg: float = 8.0
    butt_wink_rate_deg: float = 4.0

    # --- 내부 상태 (push_frame이 갱신) ---
    raw2d_buffer: list = field(default_factory=list)  # 원본(정규화 전) common-skeleton 2D
    scale2d: float | None = None
    aligned_seq: list = field(default_factory=list)  # 세션 전체, emit된(지연 적용) 정규화 3D
    emit_offset: int | None = None  # aligned_seq[0]에 해당하는 emit_idx(원본 스트림 인덱스)
    pelvis_height_hist: list = field(default_factory=list)
    pelvis_velocity_hist: list = field(default_factory=list)
    R_body: np.ndarray | None = None
    baseline_height: float | None = None
    prep_posture_ready: bool = False
    ema_raw2d_prev: np.ndarray | None = None
    prep_ground_samples: list[np.ndarray] = field(default_factory=list)
    prep_torso_samples: list[float] = field(default_factory=list)
    prep_hip_samples: list[float] = field(default_factory=list)
    ground_y: float | None = None
    baseline_ankle_ground_heights: np.ndarray | None = None
    baseline_toe_ground_heights: np.ndarray | None = None
    baseline_heel_ground_heights: np.ndarray | None = None
    baseline_torso_inclination: float | None = None
    baseline_hip_flexion: float | None = None
    heel_lift_streak: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.int32))
    heel_lift_delta: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    bottom_entry_torso: float | None = None
    bottom_entry_hip: float | None = None
    previous_torso_inclination: float | None = None
    previous_hip_flexion: float | None = None
    butt_wink_latched: bool = False

    state: str = "prep"  # prep/descend/bottom/ascend
    debounce_ctr: int = 0
    rep_start_idx: int | None = None
    phase_boundaries_running: dict = field(default_factory=dict)  # 현재 rep의 phase별 [start, end)
    current_phase_start: int = 0
    completed_reps: list = field(default_factory=list)

    def __post_init__(self):
        self.algorithm = str(self.algorithm).strip().lower()
        if self.algorithm not in DTW_ALGORITHMS:
            raise ValueError(
                f"Unsupported DTW algorithm {self.algorithm!r}; "
                f"choose one of {', '.join(DTW_ALGORITHMS)}"
            )
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1]")
        if self.heel_lift_threshold <= 0.0:
            raise ValueError("heel_lift_threshold must be positive")
        if self.weight_profile is None:
            self.weight_profile = self.weights_cfg.get("default_profile", "E_full_uniform")

        for level in DIFFICULTY_LEVELS:
            level_db = self.db_operational.get(level)
            if not isinstance(level_db, dict):
                raise ValueError(f"Operational Reference DB에 난이도가 없습니다: {level}")

            for class_label in REFERENCE_CLASSES:
                medoids = level_db.get(class_label)
                if not isinstance(medoids, (list, tuple)) or not medoids:
                    raise ValueError(
                        "Operational Reference DB 항목이 비어 있습니다: "
                        f"level={level}, class={class_label}"
                    )

                for index, medoid in enumerate(medoids):
                    meta = medoid.get("meta") if isinstance(medoid, dict) else None
                    medoid_level = meta.get("difficulty_level") if isinstance(meta, dict) else None
                    if medoid_level != level:
                        raise ValueError(
                            "Operational Reference DB 난이도 metadata가 일치하지 않습니다: "
                            f"level={level}, class={class_label}, index={index}, "
                            f"meta.difficulty_level={medoid_level!r}"
                        )

        # 고정 프레임 디바운스 대신 레퍼런스 저속 구간의 속도를 사용한다.
        self.vel_eps = self._reference_low_speed_eps()
        self.debounce_n = 0  # 호환용 필드; phase 전환에는 프레임 디바운스를 적용하지 않음

    @property
    def preparation_calibration_ready(self) -> bool:
        """운동 시작 전에 고정할 모든 준비 자세 측정값이 확보되었는지 반환한다."""
        return bool(
            self.scale2d is not None
            and self.R_body is not None
            and getattr(self, "scale3d", None) is not None
            and self.baseline_height is not None
            and self.ground_y is not None
            and self.baseline_ankle_ground_heights is not None
            and self.baseline_toe_ground_heights is not None
            and self.baseline_heel_ground_heights is not None
            and self.baseline_torso_inclination is not None
            and self.baseline_hip_flexion is not None
            and self.prep_posture_ready
        )

    def reset_preparation_calibration(self) -> None:
        """취소된 시도의 측정값을 버리고 새 카운트다운 측정을 준비한다.

        완료된 반복 결과는 유지하고 스트림, FSM, 좌표계와 준비 자세 기준만
        초기화한다.
        """
        self.raw2d_buffer.clear()
        self.scale2d = None
        self.aligned_seq.clear()
        self.emit_offset = None
        self.pelvis_height_hist.clear()
        self.pelvis_velocity_hist.clear()
        self.R_body = None
        self.scale3d = None
        self.baseline_height = None
        self.prep_posture_ready = False
        self.ema_raw2d_prev = None
        self.prep_ground_samples.clear()
        self.prep_torso_samples.clear()
        self.prep_hip_samples.clear()
        self.ground_y = None
        self.baseline_ankle_ground_heights = None
        self.baseline_toe_ground_heights = None
        self.baseline_heel_ground_heights = None
        self.baseline_torso_inclination = None
        self.baseline_hip_flexion = None
        self.heel_lift_streak.fill(0)
        self.heel_lift_delta.fill(0.0)
        self.bottom_entry_torso = None
        self.bottom_entry_hip = None
        self.previous_torso_inclination = None
        self.previous_hip_flexion = None
        self.butt_wink_latched = False
        self.state = "prep"
        self.debounce_ctr = 0
        self.rep_start_idx = None
        self.phase_boundaries_running.clear()
        self.current_phase_start = 0
        self._calib_buffer_3d = []

    def _prep_posture_ok(self, coords: np.ndarray) -> bool:
        """발-골반-어깨가 거의 일직선인 준비/완료 자세인지 확인한다."""
        shoulder = (coords[_L_SHOULDER] + coords[_R_SHOULDER]) * 0.5
        ankle = (coords[_L_ANKLE] + coords[_R_ANKLE]) * 0.5
        hip = coords[_HIP]
        foot_hip_shoulder = _angle_deg(ankle, hip, shoulder)
        # 신체 중심선은 수직에 가깝고, 양발은 같은 지면 높이에 있어야 한다.
        center_line_ok = abs(shoulder[0] - hip[0]) <= 0.25 and abs(hip[0] - ankle[0]) <= 0.25
        feet_level_ok = abs(coords[_L_HEEL, _VERTICAL_AXIS] - coords[_R_HEEL, _VERTICAL_AXIS]) <= 0.18
        feet_flat_ok = all(
            abs(coords[heel, _VERTICAL_AXIS] - coords[toe, _VERTICAL_AXIS]) <= 0.18
            for heel, toe in ((_L_HEEL, _L_BIGTOE), (_R_HEEL, _R_BIGTOE))
        )
        return bool(foot_hip_shoulder >= 155.0 and center_line_ok and feet_level_ok and feet_flat_ok)

    def _ema_filter_2d(self, raw2d_frame: np.ndarray) -> np.ndarray:
        """Causally smooth camera coordinates before 2D-to-3D lifting."""
        frame = np.asarray(raw2d_frame, dtype=np.float64)
        if frame.shape != (len(COMMON_JOINT_NAMES), 2):
            raise ValueError(
                "raw2d_frame must have shape "
                f"({len(COMMON_JOINT_NAMES)}, 2), got {frame.shape}"
            )
        if self.ema_raw2d_prev is None:
            smoothed = frame.copy()
        else:
            smoothed = self.ema_alpha * frame + (1.0 - self.ema_alpha) * self.ema_raw2d_prev
        self.ema_raw2d_prev = smoothed
        return smoothed

    @staticmethod
    def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom <= 1e-8:
            return 0.0
        cosine = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        return float(np.degrees(np.arccos(cosine)))

    def _form_angles(self, coords: np.ndarray) -> tuple[float, float]:
        """Return torso inclination and mean hip flexion in degrees."""
        torso = coords[_NECK] - coords[_HIP]
        torso_inclination = self._angle_between(torso, np.array([0.0, 1.0, 0.0]))
        hip_left = float(_angle_deg(coords[_NECK], coords[_L_HIP], coords[_L_KNEE]))
        hip_right = float(_angle_deg(coords[_NECK], coords[_R_HIP], coords[_R_KNEE]))
        return torso_inclination, (hip_left + hip_right) * 0.5

    def _capture_preparation_baseline(
        self,
        coords: np.ndarray,
        image_coords: np.ndarray,
        torso_inclination: float,
        hip_flexion: float,
        prep_posture_ok: bool,
    ) -> None:
        """Fix the preparation-pose image-ground and posture baselines once."""
        if self.state != "prep" or not prep_posture_ok or self.ground_y is not None:
            return
        foot_y = np.array(
            [
                image_coords[_L_BIGTOE, 1],
                image_coords[_R_BIGTOE, 1],
                image_coords[_L_ANKLE, 1],
                image_coords[_R_ANKLE, 1],
                image_coords[_L_HEEL, 1],
                image_coords[_R_HEEL, 1],
            ],
            dtype=np.float64,
        )
        self.prep_ground_samples.append(foot_y)
        self.prep_torso_samples.append(torso_inclination)
        self.prep_hip_samples.append(hip_flexion)
        if len(self.prep_ground_samples) < self.calib_frames:
            return

        samples = np.stack(self.prep_ground_samples[-self.calib_frames :])
        # The preparation-pose image-Y mean of toes and ankles establishes y=0.
        # Keep the existing toe/ankle ground definition for compatibility,
        # while recording the actual heel landmarks as an independent 2D
        # baseline. Eight-frame medians suppress preparation-pose jitter.
        self.ground_y = float(np.median(samples[:, :4].mean(axis=1)))
        self.baseline_ankle_ground_heights = np.median(
            samples[:, 2:4] - self.ground_y, axis=0
        )
        self.baseline_toe_ground_heights = np.median(
            samples[:, 0:2] - self.ground_y, axis=0
        )
        self.baseline_heel_ground_heights = np.median(
            samples[:, 4:6] - self.ground_y, axis=0
        )
        self.baseline_torso_inclination = float(
            np.median(self.prep_torso_samples[-self.calib_frames :])
        )
        self.baseline_hip_flexion = float(
            np.median(self.prep_hip_samples[-self.calib_frames :])
        )

    def _form_warnings(
        self,
        coords: np.ndarray,
        image_coords: np.ndarray,
        torso_inclination: float,
        hip_flexion: float,
        event: str | None,
    ) -> tuple[str, ...]:
        """Update heel-lift and bottom-position pelvic-tuck risk indicators.

        The pelvis marker alone cannot diagnose lumbar flexion.  The butt-wink
        signal is therefore deliberately a *risk proxy*: an additional abrupt
        torso/hip change after entering the bottom phase.
        """
        if event == "rep_start":
            self.bottom_entry_torso = None
            self.bottom_entry_hip = None
            self.butt_wink_latched = False

        warnings: list[str] = []
        if (
            self.ground_y is not None
            and self.baseline_toe_ground_heights is not None
            and self.baseline_heel_ground_heights is not None
        ):
            toe_ground_height = np.array(
                [
                    image_coords[_L_BIGTOE, 1] - self.ground_y,
                    image_coords[_R_BIGTOE, 1] - self.ground_y,
                ],
                dtype=np.float64,
            )
            heel_ground_height = np.array(
                [
                    image_coords[_L_HEEL, 1] - self.ground_y,
                    image_coords[_R_HEEL, 1] - self.ground_y,
                ],
                dtype=np.float64,
            )
            scale = max(float(self.scale2d or 0.0), 1e-6)
            toe_shift = (
                self.baseline_toe_ground_heights - toe_ground_height
            ) / scale
            # Camera image Y grows downward. A heel lift makes heel Y smaller.
            # Require the matching big toe to remain near its preparation
            # ground height, so whole-body/image translation is not mistaken
            # for an isolated heel lift.
            self.heel_lift_delta = (
                self.baseline_heel_ground_heights - heel_ground_height
            ) / scale
            finite = np.isfinite(self.heel_lift_delta) & np.isfinite(toe_shift)
            toe_grounded = np.abs(toe_shift) <= self.heel_lift_threshold
            lift_candidate = (
                finite
                & toe_grounded
                & (self.heel_lift_delta > self.heel_lift_threshold)
            )
            self.heel_lift_streak = np.where(lift_candidate, self.heel_lift_streak + 1, 0)
            lifted = self.heel_lift_streak >= 2
            if self.state != "prep" and bool(np.any(lifted)):
                sides = "/".join(side for side, active in zip(("L", "R"), lifted) if active)
                warnings.append(f"HEEL LIFT ({sides})")

        if self.state == "bottom":
            if self.bottom_entry_torso is None:
                self.bottom_entry_torso = torso_inclination
                self.bottom_entry_hip = hip_flexion
            elif self.baseline_torso_inclination is not None and self.baseline_hip_flexion is not None:
                torso_bottom_delta = abs(torso_inclination - self.bottom_entry_torso)
                hip_bottom_delta = abs(hip_flexion - float(self.bottom_entry_hip))
                torso_rate = abs(torso_inclination - float(self.previous_torso_inclination or torso_inclination))
                hip_rate = abs(hip_flexion - float(self.previous_hip_flexion or hip_flexion))
                changed_from_prep = (
                    abs(torso_inclination - self.baseline_torso_inclination) >= 10.0
                    or abs(hip_flexion - self.baseline_hip_flexion) >= 10.0
                )
                late_bottom_change = (
                    torso_bottom_delta >= self.butt_wink_bottom_delta_deg
                    and hip_bottom_delta >= self.butt_wink_bottom_delta_deg * 0.5
                )
                abrupt_change = (
                    torso_rate >= self.butt_wink_rate_deg
                    and hip_rate >= self.butt_wink_rate_deg * 0.5
                )
                if changed_from_prep and (late_bottom_change or abrupt_change):
                    self.butt_wink_latched = True

        if self.butt_wink_latched:
            warnings.append("BUTT WINK RISK")
        self.previous_torso_inclination = torso_inclination
        self.previous_hip_flexion = hip_flexion
        return tuple(warnings)

    def _heel_contact_evidence(
        self, start_t: int, end_t: int
    ) -> tuple[bool | None, tuple[float, float] | None]:
        """Return robust 2D heel-contact evidence for one completed repetition.

        ``True`` means both heel landmarks stayed near their preparation
        heights while the corresponding big toes stayed on the preparation
        ground. ``False`` means an upward heel displacement persisted for at
        least two frames. ``None`` means the coordinate evidence was
        insufficient or contradictory.

        Common-skeleton 2D frames do not carry landmark confidence. We
        therefore only accept finite samples, anchor each heel to its same-side
        toe, require at least half of the repetition to be usable, and never
        turn this negative evidence into a normal-pose decision by itself.
        """
        if (
            self.ground_y is None
            or self.baseline_toe_ground_heights is None
            or self.baseline_heel_ground_heights is None
            or self.scale2d is None
            or not self.raw2d_buffer
        ):
            return None, None

        start = max(0, int(start_t))
        end = min(int(end_t), len(self.raw2d_buffer) - 1)
        if end < start:
            return None, None

        return evaluate_heel_contact_2d(
            np.asarray(self.raw2d_buffer[start : end + 1], dtype=np.float64),
            scale=float(self.scale2d),
            baseline_toe_y=self.ground_y + self.baseline_toe_ground_heights,
            baseline_heel_y=self.ground_y + self.baseline_heel_ground_heights,
            threshold=self.heel_lift_threshold,
        )

    def _descending_speed_increasing(self, velocity: float) -> bool:
        if len(self.pelvis_velocity_hist) < 2:
            return True
        # 높이 속도는 음수일수록 하강 속도가 커지는 것으로 본다.
        return velocity <= self.pelvis_velocity_hist[-2] + 1e-4

    def _reference_low_speed_eps(self) -> float:
        """레퍼런스 DB의 준비·최저점·완료 저속 구간에서 실시간 임계값을 산출한다."""
        estimates: list[float] = []
        for level_db in self.db_operational.values():
            if not isinstance(level_db, dict):
                continue
            for medoids in level_db.values():
                if not isinstance(medoids, (list, tuple)):
                    continue
                for medoid in medoids:
                    feat = medoid.get("feat", {}) if isinstance(medoid, dict) else {}
                    trajectory = feat.get("pelvis_trajectory") if isinstance(feat, dict) else None
                    bounds = medoid.get("bounds", {}) if isinstance(medoid, dict) else {}
                    if trajectory is None or not bounds:
                        continue
                    velocity = np.asarray(trajectory)[:, 1]
                    low_speed = []
                    # 저장 순서: 준비, 하강, 최저점, 상승, 완료
                    for phase_index in (0, 2, 4):
                        if phase_index >= len(bounds):
                            continue
                        start, end = list(bounds.values())[phase_index]
                        start = max(0, int(start))
                        end = min(len(velocity), int(end))
                        if end > start:
                            low_speed.append(np.abs(velocity[start:end]))
                    if low_speed:
                        values = np.concatenate(low_speed)
                        values = values[np.isfinite(values)]
                        if values.size:
                            estimates.append(float(np.max(values)))
        return max(float(np.median(estimates)), 1e-6) if estimates else 0.004

    # ------------------------------------------------------------------
    def push_frame(
        self, raw2d_frame: np.ndarray, *, evaluate_motion: bool = True
    ) -> dict | None:
        """raw2d_frame: (18,2) camera1 common-skeleton pixel 좌표, 이번에 새로 도착한 프레임.

        반환: 이번 호출로 새로 "확정(emit)"된 과거 프레임(있다면)의 상태 dict, 없으면 None.
        """
        self.raw2d_buffer.append(self._ema_filter_2d(raw2d_frame))
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
        with torch.no_grad():
            hip_centered_3d = self.model(x)[0].cpu().numpy()  # (18,3), 이미 Hip-centered

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

        ankle_vert = (aligned[_L_ANKLE, _VERTICAL_AXIS] + aligned[_R_ANKLE, _VERTICAL_AXIS]) / 2.0
        pelvis_height = -ankle_vert
        self.pelvis_height_hist.append(pelvis_height)
        self.pelvis_velocity_hist.append(self._causal_velocity())
        if self.baseline_height is None and self.state == "prep" and len(self.pelvis_height_hist) >= self.calib_frames:
            self.baseline_height = float(np.median(self.pelvis_height_hist[-self.calib_frames :]))

        velocity = self.pelvis_velocity_hist[-1]
        prep_posture_ok = self._prep_posture_ok(aligned)
        torso_inclination, hip_flexion = self._form_angles(aligned)
        image_coords = self.raw2d_buffer[emit_idx]
        self._capture_preparation_baseline(
            aligned, image_coords, torso_inclination, hip_flexion, prep_posture_ok
        )
        if self.state == "prep" and prep_posture_ok:
            self.prep_posture_ready = True
        event = None
        if evaluate_motion:
            event = self._update_phase_state(t, velocity, pelvis_height, prep_posture_ok)
        form_warnings = self._form_warnings(
            aligned, image_coords, torso_inclination, hip_flexion, event
        )

        partial = self._partial_online_distance(t) if evaluate_motion else None

        return {
            "status": "ok",
            "emit_frame": emit_idx,
            "session_frame": t,
            "phase": self.state,
            "pelvis_height": pelvis_height,
            "velocity": velocity,
            "event": event,
            "partial_distance": partial,
            "form_warnings": form_warnings,
            "heel_lift_delta": tuple(float(value) for value in self.heel_lift_delta),
            "ground_ready": self.ground_y is not None,
            "completed_rep": self.completed_reps[-1] if event == "rep_end" else None,
        }

    # ------------------------------------------------------------------
    def _causal_velocity(self, win: int = 5) -> float:
        h = self.pelvis_height_hist
        if len(h) < 2:
            return 0.0
        a = h[-min(win, len(h)) :]
        return float(a[-1] - a[0]) / max(1, len(a) - 1)

    def _update_phase_state(
        self, t: int, velocity: float, pelvis_height: float, prep_posture_ok: bool
    ) -> str | None:
        event = None
        if self.state == "prep":
            if velocity < -self.vel_eps and self.prep_posture_ready and self._descending_speed_increasing(velocity):
                # 레퍼런스 저속 임계값을 넘는 즉시 하강으로 전환한다.
                self.state = "descend"
                self.rep_start_idx = max(0, t)
                self.phase_boundaries_running = {"준비": [self.current_phase_start, self.rep_start_idx]}
                self.current_phase_start = self.rep_start_idx
                self.debounce_ctr = 0
                event = "rep_start"
        elif self.state == "descend":
            recent_min = min(self.pelvis_height_hist[-5:]) if self.pelvis_height_hist else pelvis_height
            if abs(velocity) <= self.vel_eps and pelvis_height <= recent_min + 0.02:
                # 저속 구간 진입 즉시 최저점으로 전환한다.
                self.phase_boundaries_running["하강"] = [self.current_phase_start, t]
                self.current_phase_start = t
                self.state = "bottom"
                self.debounce_ctr = 0
        elif self.state == "bottom":
            height_increasing = (
                len(self.pelvis_height_hist) < 2
                or pelvis_height > self.pelvis_height_hist[-2]
            )
            if velocity > self.vel_eps and height_increasing:
                # 레퍼런스 속도 방향이 상승으로 바뀌는 즉시 전환한다.
                self.phase_boundaries_running["최저점"] = [self.current_phase_start, t]
                self.current_phase_start = t
                self.state = "ascend"
                self.debounce_ctr = 0
        elif self.state == "ascend":
            near_baseline = self.baseline_height is not None and self.pelvis_height_hist[-1] >= 0.85 * self.baseline_height
            height_increasing = (
                len(self.pelvis_height_hist) < 2
                or pelvis_height >= self.pelvis_height_hist[-2]
            )
            if (abs(velocity) <= self.vel_eps or near_baseline) and prep_posture_ok and height_increasing:
                # 레퍼런스 완료 저속 구간과 기준 높이 복귀를 즉시 완료로 판정한다.
                self.phase_boundaries_running["상승"] = [self.current_phase_start, t]
                self.phase_boundaries_running["종료"] = [t, t + 1]
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
        start = self.rep_start_idx if self.rep_start_idx is not None else 0
        rep_coords = np.stack(self.aligned_seq[self._arr_idx(start) : self._arr_idx(end_t) + 1])
        feat = extract_all_features(rep_coords)
        bounds = {p: [max(0, s - start), max(0, e - start)] for p, (s, e) in self.phase_boundaries_running.items()}
        for p in PHASES:
            bounds.setdefault(p, [0, 0])

        per_level: dict[str, dict[str, dict]] = {}
        raw_distance_by_level: dict[str, dict[str, float]] = {}
        for level in DIFFICULTY_LEVELS:
            per_level[level] = {}
            raw_distance_by_level[level] = {}
            for cls in REFERENCE_CLASSES:
                medoids = self.db_operational[level][cls]
                w = resolve_weights(self.weights_cfg, self.weight_profile, class_label=cls)
                comparison = multi_reference_distance(
                    feat,
                    bounds,
                    medoids,
                    w,
                    self.weights_cfg,
                    top_k=2,
                    algorithm=self.algorithm,
                    class_label=cls,
                )
                per_level[level][cls] = comparison
                raw_distance_by_level[level][cls] = float(comparison["min_distance"])

        heel_contact_2d, heel_lift_max_delta = self._heel_contact_evidence(start, end_t)
        decision = decide_reference_match(
            per_level,
            score_calibration=self.score_calib,
            rejection_config=self.weights_cfg.get("rejection"),
            heel_contact_2d=heel_contact_2d,
        )
        selected_contrib = decision.selected_result["best_detail"]["per_feature_contrib"]
        top_feat = sorted(selected_contrib.items(), key=lambda kv: -kv[1])[:3]

        result = RepResult(
            rep_index=len(self.completed_reps),
            frame_range=(start, end_t),
            predicted_class=decision.predicted_class,
            matched_level=decision.matched_level,
            match_rate=decision.match_rate,
            normal_match_by_level=decision.normal_match_by_level,
            normal_distance_by_level=decision.normal_distance_by_level,
            raw_distance_by_level=raw_distance_by_level,
            raw_distance_by_class=decision.raw_distance_by_class,
            score_vs_normal=decision.match_rate,
            top_contributing_features=top_feat,
            decision_status=decision.decision_status,
            rejection_reason=decision.rejection_reason,
            error_relative_margin=decision.error_relative_margin,
            error_z_margin=decision.error_z_margin,
            normal_probability=decision.normal_probability,
            heel_contact_2d=heel_contact_2d,
            heel_lift_max_delta=heel_lift_max_delta,
        )
        self.completed_reps.append(result)

    def _partial_online_distance(self, t: int) -> dict | None:
        """현재 phase 진입 이후 지금까지의 partial 시퀀스를, 각 클래스 reference의
        동일 phase와 subsequence 방식(끝점 미고정)으로 비교한 distance. 실시간 모니터링용."""
        if self.state == "prep" or t - self.current_phase_start < 2:
            return None
        phase_name = {"descend": "하강", "bottom": "최저점", "ascend": "상승"}.get(self.state)
        if phase_name is None:
            return None

        partial_coords = np.stack(self.aligned_seq[self._arr_idx(self.current_phase_start) : self._arr_idx(t) + 1])
        partial_feat = extract_all_features(partial_coords)

        from .dtw_compare import weighted_frame_cost_matrix

        if self.algorithm == "ddtw":
            partial_feat = derivative_features(partial_feat)

        per_level: dict[str, dict[str, dict[str, float]]] = {}
        distance_by_level: dict[str, dict[str, float]] = {}
        for level in DIFFICULTY_LEVELS:
            per_level[level] = {}
            distance_by_level[level] = {}
            for cls in REFERENCE_CLASSES:
                medoids = self.db_operational[level][cls]
                w = resolve_weights(self.weights_cfg, self.weight_profile, class_label=cls)
                best = None
                for med in medoids:
                    s, e = med["bounds"][phase_name]
                    if e <= s:
                        continue
                    ref_feat = {k: v[s:e] for k, v in med["feat"].items()}
                    if self.algorithm == "ddtw":
                        ref_feat = derivative_features(ref_feat)
                    cost, _ = weighted_frame_cost_matrix(
                        partial_feat, ref_feat, w, self.weights_cfg
                    )
                    if cost.size == 0:
                        continue
                    # subsequence DTW: query(부분) 전체는 매칭, reference 끝점은 자유
                    d = _subsequence_dtw_last_row_min(cost)
                    if best is None or d < best:
                        best = d

                # 난이도×클래스 중 하나라도 현재 phase가 비어 있으면
                # any-level 판정을 만들 수 없으므로 이번 partial 결과는 보류한다.
                if best is None:
                    return None
                distance = float(best)
                per_level[level][cls] = {"min_distance": distance}
                distance_by_level[level][cls] = distance

        # Partial subsequence distance is on a much smaller scale than the
        # completed five-phase distance.  Reusing the completed-repetition
        # threshold makes almost every descending/ascending prefix look
        # normal, so each live phase uses its actor-disjoint calibrated value.
        # The completed repetition continues to use ``self.score_calib``
        # unchanged in ``_finalize_rep``.
        partial_score_calib = self.score_calib
        partial_threshold = self.weights_cfg.get(
            "partial_normal_distance_thresholds", {}
        ).get(phase_name)
        if partial_score_calib is not None:
            partial_score_calib = dict(partial_score_calib)
            # Full-repetition meta-classifier and error distributions are not
            # calibrated on partial subsequence distances.
            partial_score_calib.pop("binary_classifier", None)
            partial_score_calib.pop("error_distance_normalization", None)
            partial_score_calib.pop("min_error_z_margin", None)
        if partial_score_calib is not None and partial_threshold is not None:
            partial_score_calib["normal_distance_threshold"] = float(
                partial_threshold
            )

        # 정상 threshold를 먼저 적용하고, 벗어난 경우에만 가장 가까운
        # 오류 유형을 선택한다.
        decision = decide_reference_match(
            per_level,
            score_calibration=partial_score_calib,
            rejection_config=self.weights_cfg.get("rejection"),
        )
        return {
            "phase": phase_name,
            "predicted_class": decision.predicted_class,
            "matched_level": decision.matched_level,
            "distance_by_level": distance_by_level,
            "distance_by_class": decision.raw_distance_by_class,
            "decision_status": decision.decision_status,
            "rejection_reason": decision.rejection_reason,
            "error_relative_margin": decision.error_relative_margin,
            "error_z_margin": decision.error_z_margin,
            "normal_probability": decision.normal_probability,
            "normal_distance_threshold": partial_threshold,
        }


def _subsequence_dtw_last_row_min(cost: np.ndarray) -> float:
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, :] = 0.0  # reference 시작점 자유 (subsequence)
    for i in range(1, n + 1):
        row = cost[i - 1]
        for j in range(1, m + 1):
            d[i, j] = row[j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, :].min()) / n  # reference 끝점도 자유
