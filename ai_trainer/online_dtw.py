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
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .common_skeleton import COMMON_JOINT_NAMES
from .dtw_compare import PHASES, multi_reference_distance, resolve_weights
from .features import extract_all_features
from .lifting_dataset import WINDOW_T
from .normalization import body_axes, hip_center_3d, leg_length_scale

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_L_ANKLE, _R_ANKLE = _IDX["LAnkle"], _IDX["RAnkle"]
_VERTICAL_AXIS = 1
_HALF = WINDOW_T // 2  # =4


@dataclass
class RepResult:
    rep_index: int
    frame_range: tuple[int, int]
    predicted_class: str
    raw_distance_by_class: dict[str, float]
    score_vs_normal: float | None
    top_contributing_features: list[tuple[str, float]]


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

    # --- 내부 상태 (push_frame이 갱신) ---
    raw2d_buffer: list = field(default_factory=list)  # 원본(정규화 전) common-skeleton 2D
    scale2d: float | None = None
    aligned_seq: list = field(default_factory=list)  # 세션 전체, emit된(지연 적용) 정규화 3D
    emit_offset: int | None = None  # aligned_seq[0]에 해당하는 emit_idx(원본 스트림 인덱스)
    pelvis_height_hist: list = field(default_factory=list)
    R_body: np.ndarray | None = None
    baseline_height: float | None = None

    state: str = "prep"  # prep/descend/bottom/ascend
    debounce_ctr: int = 0
    rep_start_idx: int | None = None
    phase_boundaries_running: dict = field(default_factory=dict)  # 현재 rep의 phase별 [start, end)
    current_phase_start: int = 0
    completed_reps: list = field(default_factory=list)

    def __post_init__(self):
        if self.weight_profile is None:
            self.weight_profile = self.weights_cfg.get("default_profile", "E_full_uniform")

    # ------------------------------------------------------------------
    def push_frame(self, raw2d_frame: np.ndarray) -> dict | None:
        """raw2d_frame: (18,2) camera1 common-skeleton pixel 좌표, 이번에 새로 도착한 프레임.

        반환: 이번 호출로 새로 "확정(emit)"된 과거 프레임(있다면)의 상태 dict, 없으면 None.
        """
        self.raw2d_buffer.append(raw2d_frame)
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
        if self.baseline_height is None and self.state == "prep" and len(self.pelvis_height_hist) >= self.calib_frames:
            self.baseline_height = float(np.median(self.pelvis_height_hist[-self.calib_frames :]))

        velocity = self._causal_velocity()
        event = self._update_phase_state(t, velocity)

        partial = self._partial_online_distance(t)

        return {
            "status": "ok",
            "emit_frame": emit_idx,
            "session_frame": t,
            "phase": self.state,
            "pelvis_height": pelvis_height,
            "velocity": velocity,
            "event": event,
            "partial_distance": partial,
            "completed_rep": self.completed_reps[-1] if event == "rep_end" else None,
        }

    # ------------------------------------------------------------------
    def _causal_velocity(self, win: int = 5) -> float:
        h = self.pelvis_height_hist
        if len(h) < 2:
            return 0.0
        a = h[-min(win, len(h)) :]
        return float(a[-1] - a[0]) / max(1, len(a) - 1)

    def _update_phase_state(self, t: int, velocity: float) -> str | None:
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
        start = self.rep_start_idx if self.rep_start_idx is not None else 0
        rep_coords = np.stack(self.aligned_seq[self._arr_idx(start) : self._arr_idx(end_t) + 1])
        feat = extract_all_features(rep_coords)
        bounds = {p: [max(0, s - start), max(0, e - start)] for p, (s, e) in self.phase_boundaries_running.items()}
        for p in PHASES:
            bounds.setdefault(p, [0, 0])

        per_class = {}
        for cls, medoids in self.db_operational.items():
            w = resolve_weights(self.weights_cfg, self.weight_profile, class_label=cls)
            per_class[cls] = multi_reference_distance(feat, bounds, medoids, w, self.weights_cfg, top_k=2)
        pred = min(per_class, key=lambda c: per_class[c]["min_distance"])
        score = None
        if self.score_calib is not None:
            from .scoring import distance_to_score

            score = distance_to_score(per_class["정상"]["min_distance"], self.score_calib)
        top_feat = sorted(per_class[pred]["best_detail"]["per_feature_contrib"].items(), key=lambda kv: -kv[1])[:3]

        result = RepResult(
            rep_index=len(self.completed_reps),
            frame_range=(start, end_t),
            predicted_class=pred,
            raw_distance_by_class={c: per_class[c]["min_distance"] for c in per_class},
            score_vs_normal=score,
            top_contributing_features=top_feat,
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
        return {"phase": phase_name, "distance_by_class": out} if out else None


def _subsequence_dtw_last_row_min(cost: np.ndarray) -> float:
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, :] = 0.0  # reference 시작점 자유 (subsequence)
    for i in range(1, n + 1):
        row = cost[i - 1]
        for j in range(1, m + 1):
            d[i, j] = row[j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, :].min()) / n  # reference 끝점도 자유
