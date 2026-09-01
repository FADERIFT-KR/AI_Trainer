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
from .dtw_compare import PHASES, multi_reference_distance, resolve_weights
from .features import extract_all_features
from .lifting_dataset import WINDOW_T
from .normalization import body_axes, hip_center_3d, leg_length_scale

_IDX = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}
_L_ANKLE, _R_ANKLE = _IDX["LAnkle"], _IDX["RAnkle"]
_VERTICAL_AXIS = 1
_HALF = WINDOW_T // 2  # =4

# phase 진입 직후 이 프레임 수 미만인 partial 시퀀스는 클래스 판정에 쓰지 않는다.
# 2~3프레임짜리 스니펫으로 subsequence DTW를 하면 노이즈에 취약해 하강/상승 "진입
# 순간"마다 판정이 튀는 문제(실사용 확인: 엉덩이하방오류/발뒤꿈치오류 오탐 빈발)가 있었다.
MIN_PARTIAL_FRAMES = 5


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

        return self._process_3d_frame(hip_centered_3d, emit_idx)

    # ------------------------------------------------------------------
    def push_frame_3d(self, hip_centered_3d_frame: np.ndarray) -> dict | None:
        """hip_centered_3d_frame: (18,3) 이미 3D로 복원된, Hip-centered common-skeleton 좌표
        (예: MediaPipe `world_landmarks` 기반 `CommonSkeleton3DBridge` 출력). `push_frame`과
        달리 2D 버퍼링/윈도우/lifting 모델 추론을 전혀 거치지 않고 매 프레임 즉시 처리한다
        — lifting 모델의 4프레임 지연도 없다.

        자체 학습한 lifting 모델(AI Hub camera1 단일 카메라로만 학습)이 실제 아이폰 촬영
        영상에서 학습 분포 밖 체형/팔자세/화각을 만나면 스쿼트 깊이를 심하게 과소평가하는
        것이 확인되어(2026-08-28), 실시간 파이프라인은 이 경로로 MediaPipe 자체의 3D
        추정치(훨씬 크고 다양한 데이터로 학습됨, 같은 상황에서 깊이를 정확히 잡아냄)를
        직접 사용한다. AI Hub CSV 기반 오프라인 평가/합성 테스트는 원본 데이터가 2D
        좌표뿐이라 여전히 `push_frame`(lifting 모델 경로)을 쓴다.
        """
        # emit_idx: push_frame과 동일하게 "이 세션에서 몇 번째로 처리된 프레임인가"를 뜻하는
        # 인덱스로 통일한다(단, 여기선 지연이 없으므로 원본 스트림 인덱스와 그대로 같다).
        emit_idx = getattr(self, "_frame_3d_count", 0)
        self._frame_3d_count = emit_idx + 1
        return self._process_3d_frame(hip_centered_3d_frame, emit_idx)

    def _process_3d_frame(self, hip_centered_3d: np.ndarray, emit_idx: int) -> dict | None:
        """push_frame/push_frame_3d 공통 처리: scale/orientation 캘리브레이션(1회) ->
        body-centered 정렬 -> phase 상태기계 -> partial/joint DTW -> 상태 dict 반환."""
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

                # 캘리브레이션에 쓴 프레임들(= "준비" 자세로 서 있다고 가정하는 구간) 자체에서
                # 곧바로 baseline_height(상승 완료 판정 기준)를 계산한다. 예전엔 R_body 계산
                # 이후 "prep" 상태에서 calib_frames개를 또 새로 모아야 했는데, 카운트다운이
                # 끝나자마자 바로 하강을 시작하는 실사용 영상에서는 그 추가 수집이 끝나기
                # 전에 하강이 시작돼 baseline_height가 끝내 None으로 남았다 — 그 결과
                # "상승"에서 near_baseline 조건이 절대 충족되지 않아 REP가 영원히 끝나지
                # 않는 문제가 있었다(실제 아이폰 촬영 영상으로 재현·확인).
                calib_aligned = np.einsum("ij,fpj->fpi", self.R_body.T, scaled0)
                ankle_vert_calib = (
                    calib_aligned[:, _L_ANKLE, _VERTICAL_AXIS] + calib_aligned[:, _R_ANKLE, _VERTICAL_AXIS]
                ) / 2.0
                self.baseline_height = float(np.median(-ankle_vert_calib))
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

        velocity = self._causal_velocity()
        event = self._update_phase_state(t, velocity)

        partial = self._partial_online_distance(t)
        joint_feedback = self._joint_feedback_frame(t)

        return {
            "status": "ok",
            "emit_frame": emit_idx,
            "session_frame": t,
            "phase": self.state,
            "pelvis_height": pelvis_height,
            "velocity": velocity,
            "event": event,
            "partial_distance": partial,
            "joint_feedback": joint_feedback,
            "completed_rep": self.completed_reps[-1] if event == "rep_end" else None,
            # Hip-center+Scale+Orientation 정규화까지 끝난 (18,3) 3D 좌표. DTW 계산에
            # 이미 쓰는 것을 그대로 노출 — 화면에 "내 3D 스켈레톤"을 레퍼런스와 같은
            # 좌표계/스케일로 나란히 그려주기 위함(game_ui.screens 참고).
            "aligned_frame": aligned,
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
        if self.state == "prep" or t - self.current_phase_start < MIN_PARTIAL_FRAMES:
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

    def _joint_feedback_frame(self, t: int) -> dict | None:
        """joint_feedback.py(관절별 위치/각도 오차)가 쓸 "현재 프레임 정렬"을 찾는다.

        DTW와 관절 피드백의 역할을 분리한다는 요구사항에 따라, 여기서는 오차 계산은
        하지 않고 오직 "지금 이 순간(t)에 대응하는 '정상' reference 프레임이 무엇인가"만
        정한다 — `_partial_online_distance`와 동일한 subsequence DTW(끝점 자유)를 쓰되,
        distance 최솟값이 아니라 그 최솟값을 만든 reference 프레임 인덱스(정렬 결과)를
        반환하도록 확장한 버전. 여러 medoid 중 지금까지의 partial 시퀀스와 가장 잘
        맞는(distance가 가장 작은) medoid를 골라 그 정렬을 사용한다.
        """
        if self.state == "prep" or t - self.current_phase_start < 2:
            return None
        phase_name = {"descend": "하강", "bottom": "최저점", "ascend": "상승"}.get(self.state)
        if phase_name is None:
            return None
        medoids = self.db_operational.get("정상")
        if not medoids:
            return None

        partial_coords = np.stack(self.aligned_seq[self._arr_idx(self.current_phase_start) : self._arr_idx(t) + 1])
        partial_feat = extract_all_features(partial_coords)

        from .dtw_compare import weighted_frame_cost_matrix

        w = resolve_weights(self.weights_cfg, self.weight_profile, class_label="정상")
        best: tuple[float, int, dict, int] | None = None  # (dist, j_star, medoid, phase_seg_start)
        for med in medoids:
            s, e = med["bounds"].get(phase_name, (0, 0))
            if e <= s:
                continue
            ref_feat = {k: v[s:e] for k, v in med["feat"].items()}
            cost, _ = weighted_frame_cost_matrix(partial_feat, ref_feat, w, self.weights_cfg)
            if cost.size == 0:
                continue
            dist, j_star = _subsequence_dtw_end_index(cost)
            if best is None or dist < best[0]:
                best = (dist, j_star, med, s)

        if best is None:
            return None
        _, j_star, med, seg_start = best
        # joint_coords_3d는 features.extract_all_features가 (T,18,3)을 (T,54)로 펼친 것
        # (이미 Hip-center+Scale+Orientation 정규화 완료 좌표) — 다시 (18,3)으로 접는다.
        ref_frame = med["feat"]["joint_coords_3d"][seg_start + j_star].reshape(-1, 3)
        user_frame = self.aligned_seq[self._arr_idx(t)]
        return {"phase": phase_name, "medoid": med["meta"], "user_frame": user_frame, "ref_frame": ref_frame}


def _subsequence_dtw_last_row_min(cost: np.ndarray) -> float:
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, :] = 0.0  # reference 시작점 자유 (subsequence)
    for i in range(1, n + 1):
        row = cost[i - 1]
        for j in range(1, m + 1):
            d[i, j] = row[j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, :].min()) / n  # reference 끝점도 자유


def _subsequence_dtw_end_index(cost: np.ndarray) -> tuple[float, int]:
    """`_subsequence_dtw_last_row_min`과 동일한 subsequence DTW(끝점 자유)이지만,
    최솟값뿐 아니라 그 최솟값을 만든 reference 프레임 인덱스(0-indexed, `cost`의
    열 기준)도 함께 반환한다 — "지금 마지막 query 프레임이 reference의 몇 번째
    프레임에 대응하는가"라는 정렬 정보가 필요한 joint_feedback용."""
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, :] = 0.0
    for i in range(1, n + 1):
        row = cost[i - 1]
        for j in range(1, m + 1):
            d[i, j] = row[j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    last_row = d[n, 1:]  # j=0(빈 매칭)은 항상 inf라 제외해도 결과는 같음
    j_star = int(np.argmin(last_row))
    return float(last_row[j_star]) / n, j_star
