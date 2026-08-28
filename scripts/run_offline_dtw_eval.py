#!/usr/bin/env python3
"""Offline Weighted DTW 평가 파이프라인.

Validation actor(configs/actor_split.json)의 시퀀스를 "실제 웹캠처럼" camera1 2D ->
학습된 Lifting 모델 -> 정규화 파이프라인을 거친 쿼리로 만들고, Reference DB(GT/Operational
두 tier)와 phase-aware weighted DTW로 비교해 분류/점수/피드백을 평가한다.

actor-disjoint 확인: Reference DB의 16개 medoid는 전부 train-actor 시퀀스(직전 단계에서
확인)이며, 여기서 쓰는 query는 전부 val-actor 시퀀스이므로 leakage가 없다.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.actor_split import load_all_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.dtw_compare import PHASES, multi_reference_distance, resolve_weights  # noqa: E402
from ai_trainer.features import extract_all_features  # noqa: E402
from ai_trainer.lifting_dataset import load_actor_split  # noqa: E402
from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.phase_segmentation import segment_phases  # noqa: E402
from ai_trainer.reference_db_io import load_reference_db  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference, build_operational_reference  # noqa: E402
from ai_trainer.scoring import distance_to_score, fit_score_calibration  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Training/02.라벨링데이터/TL.zip"
)
VL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"
)
SPLIT_PATH = ROOT / "configs" / "actor_split.json"
WEIGHTS_CFG_PATH = ROOT / "configs" / "dtw_feature_weights.json"
DB_DIR = ROOT / "output" / "reference_db"
OUT_DIR = ROOT / "output" / "dtw_eval"
CLASSES = ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"]


def build_query(z: AiHubZip, seq, origin: str, model, device) -> dict | None:
    ref = build_operational_reference(z, seq, origin, model, device)
    if ref is None or ref.coords.shape[0] < 10:
        return None
    feat = extract_all_features(ref.coords)
    pf = extract_phase_features(ref.coords)
    bounds = segment_phases(pf).as_dict()
    return {
        "feat": feat,
        "bounds": bounds,
        "meta": {
            "actor": seq.actor,
            "level": seq.level,
            "true_class": seq.error_type,
            "rep": seq.rep,
            "origin": origin,
            "frame_range": list(ref.frame_range),
        },
    }


def build_query_gt(z: AiHubZip, seq, origin: str) -> dict | None:
    """build_query와 동일하지만 lifting 모델을 거치지 않고 8카메라 삼각측량 실측 3D를
    그대로 쓴다 — 실시간 파이프라인이 이제 자체 lifting 모델 대신 MediaPipe 자체 3D
    (world_landmarks)를 쓰므로(2026-08-28, 학습 분포 밖 체형/화각에서 lifting 모델이
    깊이를 심하게 과소평가하는 문제 확인), score_calib도 "실제 3D 대 실제 3D" 도메인으로
    맞춰야 % 점수가 의미 있다. AI Hub는 CSV만 쓰므로 MediaPipe 자체를 여기서 돌릴 순
    없지만, 둘 다 "정확한 3D"라는 점에서 lifting 모델 도메인보다는 훨씬 가깝다."""
    ref = build_ground_truth_reference(z, seq, origin)
    if ref is None or ref.coords.shape[0] < 10:
        return None
    feat = extract_all_features(ref.coords)
    pf = extract_phase_features(ref.coords)
    bounds = segment_phases(pf).as_dict()
    return {
        "feat": feat,
        "bounds": bounds,
        "meta": {
            "actor": seq.actor,
            "level": seq.level,
            "true_class": seq.error_type,
            "rep": seq.rep,
            "origin": origin,
            "frame_range": list(ref.frame_range),
        },
    }


def classify(query: dict, db_tier: dict, weights_cfg: dict, weight_profile: str, top_k: int = 2) -> dict:
    per_class = {}
    for cls in CLASSES:
        medoids = db_tier[cls]
        w = resolve_weights(weights_cfg, weight_profile, class_label=cls)
        per_class[cls] = multi_reference_distance(query["feat"], query["bounds"], medoids, w, weights_cfg, top_k=top_k)
    return per_class


def confusion_and_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    classes = CLASSES
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    metrics = {}
    for c in classes:
        i = idx[c]
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[c] = {"precision": precision, "recall": recall, "f1": f1, "support": int(cm[i, :].sum())}

    accuracy = float(np.trace(cm)) / max(1, cm.sum())
    return {"confusion_matrix": cm.tolist(), "classes": classes, "per_class": metrics, "accuracy": accuracy}


def binary_normal_vs_error(y_true: list[str], y_pred: list[str]) -> float:
    correct = 0
    for t, p in zip(y_true, y_pred):
        t_bin = t == "정상"
        p_bin = p == "정상"
        correct += int(t_bin == p_bin)
    return correct / max(1, len(y_true))


def main() -> None:
    t_start = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TemporalLiftingNet(n_joints=18, hidden=128)
    model.load_state_dict(torch.load(ROOT / "output" / "lifting_baseline" / "model_best.pt", map_location=device))
    model.to(device).eval()

    weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
    db = load_reference_db(DB_DIR)  # db[tier][class] = [...]
    medoid_actor_ids = {m["meta"]["actor_id"] for tier in db.values() for cls in tier.values() for m in cls}
    print(f"Reference DB medoid actor: {sorted(medoid_actor_ids)}")

    actor_to_split = load_actor_split(SPLIT_PATH)
    all_seqs = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    val_seqs = [os_ for os_ in all_seqs if actor_to_split.get(os_.seq.actor) == "val" and os_.seq.error_type in CLASSES]
    print(f"Validation query 대상: {len(val_seqs)}개 (actor {len({o.seq.actor for o in val_seqs})}명)")

    overlap = medoid_actor_ids & {o.seq.actor for o in val_seqs}
    assert not overlap, f"actor leakage 발견: {overlap}"
    print("actor-disjoint 확인 통과 (medoid actor ∩ val actor = ∅)")

    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}

    # ---- 1) validation query 구축 (한 번만 계산, 이후 여러 설정에 재사용) ----
    print("\nValidation query 생성 중 (camera1 2D -> lifting model -> 정규화)...")
    t0 = time.time()
    queries = []
    for os_ in val_seqs:
        q = build_query(zips[os_.origin], os_.seq, os_.origin, model, device)
        if q is not None:
            queries.append(q)
    print(f"쿼리 {len(queries)}개 생성 완료 ({time.time()-t0:.1f}초)")

    # ---- 2) 메인 평가: tier x method 그리드 (weight_profile = D_full_weighted) ----
    print("\n메인 평가: GT/Operational tier x min/topk method 비교 중...")
    t0 = time.time()
    raw_results = defaultdict(list)  # (tier) -> [ {per_class dist dict, meta} ]
    for tier in ["ground_truth", "operational"]:
        for q in queries:
            per_class = classify(q, db[tier], weights_cfg, "D_full_weighted", top_k=2)
            raw_results[tier].append({"per_class": per_class, "meta": q["meta"]})
    print(f"완료 ({time.time()-t0:.1f}초)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {"n_queries": len(queries)}

    for tier in ["ground_truth", "operational"]:
        report[tier] = {}
        for method in ["min_distance", "mean_topk_distance"]:
            y_true = [r["meta"]["true_class"] for r in raw_results[tier]]
            y_pred = [min(CLASSES, key=lambda c: r["per_class"][c][method]) for r in raw_results[tier]]
            m = confusion_and_metrics(y_true, y_pred)
            m["binary_normal_vs_error_acc"] = binary_normal_vs_error(y_true, y_pred)

            # difficulty별 accuracy
            by_level = defaultdict(lambda: [0, 0])
            for r, t, p in zip(raw_results[tier], y_true, y_pred):
                lvl = r["meta"]["level"]
                by_level[lvl][1] += 1
                by_level[lvl][0] += int(t == p)
            m["accuracy_by_level"] = {lvl: c / n for lvl, (c, n) in by_level.items()}

            report[tier][method] = m
            print(f"[{tier}/{method}] accuracy={m['accuracy']:.3f}  binary(정상vs오류)={m['binary_normal_vs_error_acc']:.3f}")

    # ---- 3) 클래스별 DTW distance distribution (tier=operational, method=min, 정답 클래스 기준) ----
    dist_dist = defaultdict(list)
    for r in raw_results["operational"]:
        true_c = r["meta"]["true_class"]
        dist_dist[true_c].append(r["per_class"][true_c]["min_distance"])
    report["distance_distribution_operational_min"] = {
        c: {
            "mean": float(np.mean(v)), "std": float(np.std(v)), "min": float(np.min(v)),
            "p50": float(np.percentile(v, 50)), "max": float(np.max(v)), "n": len(v),
        }
        for c, v in dist_dist.items()
    }

    # ---- 4) feature ablation + weight ablation (tier=ground_truth, method=min_distance) ----
    print("\nFeature/weight ablation (A/B/C/D/E) 진행 중...")
    t0 = time.time()
    ablation_report = {}
    for profile in [
        "A_coords_only", "B_angles_only", "C_coords_plus_angles", "D_full_weighted", "E_full_uniform",
        "F_squat_form_focus",
    ]:
        y_true, y_pred = [], []
        for q in queries:
            per_class = classify(q, db["ground_truth"], weights_cfg, profile, top_k=2)
            y_true.append(q["meta"]["true_class"])
            y_pred.append(min(CLASSES, key=lambda c: per_class[c]["min_distance"]))
        m = confusion_and_metrics(y_true, y_pred)
        m["binary_normal_vs_error_acc"] = binary_normal_vs_error(y_true, y_pred)
        ablation_report[profile] = {"accuracy": m["accuracy"], "binary_acc": m["binary_normal_vs_error_acc"], "per_class_f1": {c: m["per_class"][c]["f1"] for c in CLASSES}}
        print(f"  {profile:22s} accuracy={m['accuracy']:.3f}  binary={m['binary_normal_vs_error_acc']:.3f}")
    report["ablation"] = ablation_report
    print(f"ablation 완료 ({time.time()-t0:.1f}초)")

    # ---- 5) score calibration (train 시퀀스, medoid 제외) ----
    print("\nScore calibration용 train 표본 생성 중...")
    t0 = time.time()
    train_seqs = [os_ for os_ in all_seqs if actor_to_split.get(os_.seq.actor) == "train" and os_.seq.error_type in CLASSES]
    medoid_keys = {(m["meta"]["actor_id"], m["meta"]["repetition_id"]) for m in db["ground_truth"]["정상"] + db["operational"]["정상"]}
    for cls in CLASSES:
        medoid_keys |= {(m["meta"]["actor_id"], m["meta"]["repetition_id"]) for m in db["ground_truth"][cls]}
    import random

    rng = random.Random(23)
    calib_candidates = defaultdict(list)
    for os_ in train_seqs:
        key = (os_.seq.actor, os_.seq.rep)
        if key in medoid_keys:
            continue
        calib_candidates[os_.seq.error_type].append(os_)
    calib_seqs = []
    for cls in CLASSES:
        pool = calib_candidates[cls]
        rng.shuffle(pool)
        calib_seqs.extend(pool[:25])

    # 실서비스(OnlineSquatSession)가 실제로 쓰는 weight_profile로 캘리브레이션해야
    # score_calib의 lo/hi(distance 스케일)이 앱에서 계산되는 distance와 맞는다.
    # 예전엔 "D_full_weighted"로 고정되어 있었는데, 그 뒤 default_profile이
    # F_squat_form_focus로 바뀌면서(팔 관절 raw 좌표/속도 제외) distance 스케일 자체가
    # 달라져 그대로 두면 % 점수가 틀어진다.
    calib_profile = weights_cfg["default_profile"]
    normal_d, error_d = [], []
    for os_ in calib_seqs:
        q = build_query(zips[os_.origin], os_.seq, os_.origin, model, device)
        if q is None:
            continue
        w = resolve_weights(weights_cfg, calib_profile, class_label="정상")
        r = multi_reference_distance(q["feat"], q["bounds"], db["operational"]["정상"], w, weights_cfg, top_k=2)
        (normal_d if q["meta"]["true_class"] == "정상" else error_d).append(r["min_distance"])
    calib = fit_score_calibration(np.array(normal_d), np.array(error_d))
    report["score_calibration"] = {**calib, "n_normal": len(normal_d), "n_error": len(error_d), "weight_profile": calib_profile}
    print(f"calibration 완료 ({time.time()-t0:.1f}초): lo={calib['lo']:.3f} hi={calib['hi']:.3f}  (n_normal={len(normal_d)}, n_error={len(error_d)})")

    # ---- 5b) score calibration (ground_truth tier 버전) ----
    # 실시간 파이프라인이 자체 lifting 모델 대신 MediaPipe 자체 3D(world_landmarks)를 쓰도록
    # 바뀌면서(2026-08-28), 그 경로는 db["operational"]이 아니라 db["ground_truth"]와
    # 비교한다 — 위 calibration(operational 도메인)을 그대로 쓰면 % 점수가 틀어지므로
    # ground_truth 도메인 전용 calibration을 별도로 만든다. 쿼리도 lifting 모델을 거치지
    # 않은 실측 3D(build_query_gt)로 만들어 도메인을 맞춘다.
    print("\nScore calibration(ground_truth tier)용 계산 중...")
    t0 = time.time()
    normal_d_gt, error_d_gt = [], []
    for os_ in calib_seqs:
        q = build_query_gt(zips[os_.origin], os_.seq, os_.origin)
        if q is None:
            continue
        w = resolve_weights(weights_cfg, calib_profile, class_label="정상")
        r = multi_reference_distance(q["feat"], q["bounds"], db["ground_truth"]["정상"], w, weights_cfg, top_k=2)
        (normal_d_gt if q["meta"]["true_class"] == "정상" else error_d_gt).append(r["min_distance"])
    calib_gt = fit_score_calibration(np.array(normal_d_gt), np.array(error_d_gt))
    report["score_calibration_ground_truth"] = {
        **calib_gt, "n_normal": len(normal_d_gt), "n_error": len(error_d_gt), "weight_profile": calib_profile,
    }
    print(f"calibration(GT) 완료 ({time.time()-t0:.1f}초): lo={calib_gt['lo']:.3f} hi={calib_gt['hi']:.3f}  (n_normal={len(normal_d_gt)}, n_error={len(error_d_gt)})")

    # ---- 6) 예시 몇 개 (predicted class / raw distance / 주요 error feature) ----
    print("\n예시 Validation 쿼리:")
    examples = []
    rng2 = np.random.default_rng(5)
    sample_idx = rng2.choice(len(queries), size=min(8, len(queries)), replace=False)
    for i in sample_idx:
        q = queries[i]
        per_class = classify(q, db["operational"], weights_cfg, "D_full_weighted", top_k=2)
        pred = min(CLASSES, key=lambda c: per_class[c]["min_distance"])
        score = distance_to_score(per_class["정상"]["min_distance"], calib)
        top_features = sorted(
            per_class[pred]["best_detail"]["per_feature_contrib"].items(), key=lambda kv: -kv[1]
        )[:3]
        ex = {
            "actor": q["meta"]["actor"], "level": q["meta"]["level"], "true_class": q["meta"]["true_class"],
            "predicted_class": pred, "distance_by_class": {c: per_class[c]["min_distance"] for c in CLASSES},
            "score_vs_normal": score, "top_contributing_features": top_features,
        }
        examples.append(ex)
        print(
            f"  {ex['actor']}({ex['level']}) true={ex['true_class']:8s} pred={ex['predicted_class']:8s} "
            f"score={score:5.1f}  top_feat={[f[0] for f in top_features]}"
        )
    report["examples"] = examples

    report["elapsed_sec"] = time.time() - t_start
    (OUT_DIR / "offline_eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n전체 완료 ({report['elapsed_sec']:.1f}초). 저장: {OUT_DIR / 'offline_eval_report.json'}")

    for z in zips.values():
        z.close()


if __name__ == "__main__":
    main()
