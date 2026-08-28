"""Phase-aware Weighted DTW 비교 엔진.

cost(t,r) = sum_k w_k * d_k(feature_k(t), feature_k(r))

phase(준비/하강/최저점/상승/종료)별로 독립적으로 DTW를 계산한 뒤
D_total = sum_p phase_weight[p] * D_phase[p] 로 합산한다.
시퀀스 길이가 phase마다 다르므로, phase별 DTW distance는 정렬 경로 길이로
나눠 정규화해 짧은/긴 phase 사이의 스케일을 맞춘다.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .features import FEATURE_NAMES

PHASES = ["준비", "하강", "최저점", "상승", "종료"]


def load_dtw_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# class_overrides를 덮어쓰는(=클래스별 판별 특징을 강조하는) "운영용" 프로파일.
# D는 원래부터 그랬고, F(실사용 기본값)도 포함해야 한다 — 안 그러면 모든 클래스가 동일한
# 가중치로 비교되어(예: 발뒤꿈치오류 판별에 heel_height/ankle_angle이 거의 반영 안 됨),
# 실제로는 무관한 공통 스쿼트 형태(깊이/템포) 유사성만으로 오분류가 난다(실사용 확인:
# 하강/상승 중 엉덩이하방오류·발뒤꿈치오류 오탐 빈발). E/A/B/C는 ablation 비교 목적이므로
# override를 적용하지 않고 고정 유지한다.
_CLASS_OVERRIDE_PROFILES = {"D_full_weighted", "F_squat_form_focus"}


def resolve_weights(config: dict, weight_profile: str, class_label: str | None) -> dict[str, float]:
    """weight_profile: 'D_full_weighted' | 'F_squat_form_focus' | 'E_full_uniform' | 'A_coords_only' | 'B_angles_only' | 'C_coords_plus_angles'."""
    if weight_profile == "D_full_weighted":
        w = dict(config["base_weights"])
    elif weight_profile == "E_full_uniform":
        return dict(config["uniform_weights"])
    else:
        ablation = config["ablation_configs"].get(weight_profile)
        if not isinstance(ablation, dict):
            raise ValueError(f"알 수 없는 weight_profile: {weight_profile}")
        w = {name: ablation.get(name, 0.0) for name in FEATURE_NAMES}

    if weight_profile in _CLASS_OVERRIDE_PROFILES and class_label and class_label in config["class_overrides"]:
        w.update(config["class_overrides"][class_label])
    return w


def _dtw_dp(cost: np.ndarray) -> tuple[float, int]:
    """cost: (n,m) 프레임쌍 비용행렬 -> (누적 최소비용, 정렬 경로 길이 근사(n+m))."""
    n, m = cost.shape
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0.0
    for i in range(1, n + 1):
        row_cost = cost[i - 1]
        for j in range(1, m + 1):
            d[i, j] = row_cost[j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return float(d[n, m]), n + m


def phase_slice(features: dict[str, np.ndarray], bounds: dict, phase: str) -> dict[str, np.ndarray]:
    s, e = bounds[phase]
    return {name: arr[s:e] for name, arr in features.items()}


def weighted_frame_cost_matrix(
    feat_a: dict[str, np.ndarray], feat_b: dict[str, np.ndarray], weights: dict[str, float], config: dict
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """반환: (합산 cost 행렬, {feature_name: weight*거리행렬} — 기여도 분석용)."""
    n = next(iter(feat_a.values())).shape[0]
    m = next(iter(feat_b.values())).shape[0]
    total = np.zeros((n, m))
    per_feature: dict[str, np.ndarray] = {}
    for name in FEATURE_NAMES:
        w = weights.get(name, 0.0)
        if w == 0.0 or n == 0 or m == 0:
            continue
        metric = config["features"][name]["metric"]
        d = cdist(feat_a[name], feat_b[name], metric=metric)
        d = np.nan_to_num(d, nan=0.0)  # cosine dist가 영벡터일 때 nan 방지
        contrib = w * d
        total += contrib
        per_feature[name] = contrib
    return total, per_feature


def phase_aware_weighted_dtw(
    feat_a: dict[str, np.ndarray],
    bounds_a: dict,
    feat_b: dict[str, np.ndarray],
    bounds_b: dict,
    weights: dict[str, float],
    config: dict,
) -> dict:
    """두 시퀀스 사이 phase별 DTW distance와 총합, feature별 기여도를 계산한다."""
    per_phase_dist: dict[str, float | None] = {}
    per_feature_contrib: dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    total = 0.0
    phase_weights = config["phase_weights"]

    for phase in PHASES:
        pa = phase_slice(feat_a, bounds_a, phase)
        pb = phase_slice(feat_b, bounds_b, phase)
        n = next(iter(pa.values())).shape[0]
        m = next(iter(pb.values())).shape[0]
        if n == 0 or m == 0:
            per_phase_dist[phase] = None
            continue
        cost, per_feature = weighted_frame_cost_matrix(pa, pb, weights, config)
        d, path_len = _dtw_dp(cost)
        d_norm = d / path_len
        per_phase_dist[phase] = d_norm
        total += phase_weights.get(phase, 1.0) * d_norm

        # feature 기여도: 해당 phase의 DTW 최적 경로 전체를 다시 추적하기보다,
        # 비용행렬 평균으로 근사(연산량을 줄이면서도 상대적 크기 비교에는 충분).
        for name, contrib in per_feature.items():
            per_feature_contrib[name] += phase_weights.get(phase, 1.0) * float(contrib.mean())

    return {"total": total, "per_phase": per_phase_dist, "per_feature_contrib": per_feature_contrib}


def multi_reference_distance(
    query_feat: dict[str, np.ndarray],
    query_bounds: dict,
    medoids: list[dict],  # [{feat, bounds, meta}]
    weights: dict[str, float],
    config: dict,
    top_k: int = 2,
) -> dict:
    """query 하나를 한 클래스의 medoid 전체와 비교, min/top-k 평균 두 방식을 모두 반환."""
    results = []
    for med in medoids:
        r = phase_aware_weighted_dtw(query_feat, query_bounds, med["feat"], med["bounds"], weights, config)
        results.append({"medoid": med["meta"], "dtw": r})

    dists = np.array([r["dtw"]["total"] for r in results])
    order = np.argsort(dists)
    best = results[order[0]]
    top_k_actual = min(top_k, len(dists))
    mean_topk = float(dists[order[:top_k_actual]].mean())

    return {
        "min_distance": float(dists[order[0]]),
        "mean_topk_distance": mean_topk,
        "best_medoid": best["medoid"],
        "best_detail": best["dtw"],
        "all": results,
    }
