#!/usr/bin/env python3
"""Generate front/left/right squat conditions from the local AI Hub dataset.

The generated JSON contains three auditable parts:

1. data-derived mapping of camera0..7 to front/back/left/right;
2. robust normal ranges learned only from training actors;
3. one-dimensional, interpretable error conditions selected on training actors
   and reported with actor-disjoint validation balanced accuracy.

The conditions are intended as evidence for the sequence classifier, not as a
stand-alone declaration that every unlisted movement is normal.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ai_trainer.squat.actor_split import load_air_squat_sequences  # noqa: E402
from ai_trainer.squat.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.squat.camera_views import VIEWS, infer_camera_views  # noqa: E402
from ai_trainer.core.s3_mapping.common_skeleton import to_common_skeleton  # noqa: E402
from ai_trainer.core.s1_capture.dataset_config import DATASET_PATH  # noqa: E402
from ai_trainer.squat.lifting_dataset import load_actor_split  # noqa: E402
from ai_trainer.squat.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.squat.phase_segmentation import segment_phases  # noqa: E402
from ai_trainer.squat.reference_pipeline import build_ground_truth_reference  # noqa: E402
from ai_trainer.squat.view_conditions import extract_view_features, summarize_view_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = ROOT / "configs" / "actor_split.json"
OUT_PATH = ROOT / "configs" / "view_condition_thresholds.json"
NORMAL_CLASS = "정상"
ERROR_CLASSES = ("발뒤꿈치오류", "엉덩이하방오류", "고관절오류")
MAX_RULES_PER_ERROR_VIEW = 4
MIN_TRAIN_BALANCED_ACCURACY = 0.64
MIN_TUNING_BALANCED_ACCURACY = 0.55


def _camera_sample(origin_sequences, max_per_group: int = 1):
    selected = []
    counts: Counter[tuple[str, str]] = Counter()
    for item in sorted(origin_sequences, key=lambda value: (value.seq.actor, value.seq.error_type, int(value.seq.rep))):
        key = (item.seq.actor[:2], item.seq.error_type)
        if counts[key] >= max_per_group:
            continue
        counts[key] += 1
        selected.append(item.seq)
    return selected


def _balanced_accuracy(values: np.ndarray, labels: np.ndarray, threshold: float, operator: str) -> float:
    predicted_error = values >= threshold if operator == ">=" else values <= threshold
    true_positive = np.mean(predicted_error[labels == 1]) if np.any(labels == 1) else 0.0
    true_negative = np.mean(~predicted_error[labels == 0]) if np.any(labels == 0) else 0.0
    return float((true_positive + true_negative) / 2.0)


def _best_threshold(normal_values: list[float], error_values: list[float]) -> dict | None:
    if len(normal_values) < 8 or len(error_values) < 8:
        return None
    values = np.asarray(normal_values + error_values, dtype=np.float64)
    labels = np.asarray([0] * len(normal_values) + [1] * len(error_values), dtype=np.int8)
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    unique = np.unique(values)
    if unique.size < 2:
        return None
    candidates = (unique[:-1] + unique[1:]) / 2.0
    if candidates.size > 300:
        indices = np.linspace(0, candidates.size - 1, 300).astype(int)
        candidates = candidates[indices]
    best = None
    for operator in (">=", "<="):
        for threshold in candidates:
            score = _balanced_accuracy(values, labels, float(threshold), operator)
            candidate = (score, -abs(float(threshold) - float(np.median(values))), operator, float(threshold))
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return None
    return {"balanced_accuracy": float(best[0]), "operator": best[2], "threshold": best[3]}


def _metric_family(metric: str) -> str:
    parts = metric.split("|")
    if parts[0] == "timing":
        return "|".join(parts[:2])
    if len(parts) < 2:
        return metric
    feature = parts[1]
    # angle and flexion are exact complements (180-angle), so they must not
    # receive separate votes.  Ignore phase/stat here as well: selected rules
    # should represent distinct biomechanical quantities, not four correlated
    # summaries of the same quantity.
    feature = feature.replace("visible_hip_angle_deg", "visible_hip_flexion")
    feature = feature.replace("visible_hip_flexion_deg", "visible_hip_flexion")
    feature = feature.replace("visible_knee_angle_deg", "visible_knee_flexion")
    feature = feature.replace("visible_knee_flexion_deg", "visible_knee_flexion")
    return feature


def _finite_json(value):
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _score_records(normal_records: list[dict], error_records: list[dict], metric: str, rule: dict) -> float:
    values = np.asarray(
        [record["metrics"].get(metric, np.nan) for record in normal_records + error_records],
        dtype=np.float64,
    )
    labels = np.asarray([0] * len(normal_records) + [1] * len(error_records), dtype=np.int8)
    finite = np.isfinite(values)
    if not finite.any() or not np.any(labels[finite] == 0) or not np.any(labels[finite] == 1):
        return 0.5
    return _balanced_accuracy(values[finite], labels[finite], rule["threshold"], rule["operator"])


def main() -> None:
    actor_split = load_actor_split(SPLIT_PATH)
    train_actors = sorted(actor for actor, split in actor_split.items() if split == "train")
    # Error-rule threshold fitting and rule selection must use disjoint actors.
    # Every fifth sorted train actor is a deterministic internal tuning split;
    # the existing val actors remain untouched until final reporting.
    rule_tune_actors = set(train_actors[4::5])
    rule_fit_actors = set(train_actors) - rule_tune_actors
    if not rule_tune_actors or not rule_fit_actors:
        raise ValueError("오류 조건용 train actor를 fit/tune으로 나눌 수 없습니다.")
    origin_sequences = load_air_squat_sequences(DATASET_PATH)
    print(f"전체 에어스쿼트 시퀀스: {len(origin_sequences)}개")

    with AiHubZip(DATASET_PATH) as dataset:
        camera_sample = _camera_sample(origin_sequences)
        mapping = infer_camera_views(dataset, camera_sample, frames_per_sequence=5)
        camera_by_view = {view: mapping.as_dict()[view] for view in VIEWS}
        print(
            "카메라 분류: "
            f"front=camera{mapping.front}, back=camera{mapping.back}, "
            f"left=camera{mapping.left}, right=camera{mapping.right}"
        )

        records: list[dict] = []
        skipped = 0
        for index, item in enumerate(origin_sequences, start=1):
            split = actor_split.get(item.seq.actor)
            if split not in {"train", "val"}:
                skipped += 1
                continue
            reference = build_ground_truth_reference(dataset, item.seq, item.origin)
            if reference is None or reference.coords.shape[0] < 8:
                skipped += 1
                continue
            phase_bounds = segment_phases(extract_phase_features(reference.coords)).as_dict()
            start, end = reference.frame_range
            for view, camera in camera_by_view.items():
                try:
                    _, coords_26 = dataset.read_2d(item.seq, camera)
                except FileNotFoundError:
                    continue
                clipped = coords_26[start : end + 1]
                n = min(len(clipped), len(reference.coords))
                if n < 8:
                    continue
                clipped = to_common_skeleton(clipped[:n])
                bounded = {
                    phase: [max(0, min(int(a), n)), max(0, min(int(b), n))]
                    for phase, (a, b) in phase_bounds.items()
                }
                summary = summarize_view_features(extract_view_features(clipped, view), bounded)
                records.append(
                    {
                        "actor": item.seq.actor,
                        "split": split,
                        "error_type": item.seq.error_type,
                        "level": item.seq.level,
                        "view": view,
                        "metrics": summary,
                    }
                )
            if index % 50 == 0:
                print(f"  ...{index}/{len(origin_sequences)} 시퀀스 처리")

    print(f"유효 view 시퀀스: {len(records)}개, 스킵: {skipped}개")
    counts = Counter((record["split"], record["error_type"], record["view"]) for record in records)

    normal_ranges: dict[str, dict[str, dict]] = {view: {} for view in VIEWS}
    for view in VIEWS:
        normal_train = [
            record for record in records
            if record["view"] == view and record["split"] == "train" and record["error_type"] == NORMAL_CLASS
        ]
        metric_names = sorted({name for record in normal_train for name in record["metrics"]})
        for metric in metric_names:
            values = np.asarray(
                [record["metrics"][metric] for record in normal_train if metric in record["metrics"]],
                dtype=np.float64,
            )
            values = values[np.isfinite(values)]
            if len(values) < 8:
                continue
            q025, q10, median, q90, q975 = np.percentile(values, [2.5, 10, 50, 90, 97.5])
            normal_ranges[view][metric] = {
                "hard_lower": round(float(q025), 6),
                "warning_lower": round(float(q10), 6),
                "median": round(float(median), 6),
                "warning_upper": round(float(q90), 6),
                "hard_upper": round(float(q975), 6),
                "iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 6),
                "n": int(len(values)),
            }

    error_rules: dict[str, dict[str, list[dict]]] = {
        view: {error: [] for error in ERROR_CLASSES} for view in VIEWS
    }
    for view in VIEWS:
        for error_class in ERROR_CLASSES:
            fit_normal = [
                record for record in records
                if record["view"] == view and record["actor"] in rule_fit_actors
                and record["error_type"] == NORMAL_CLASS
            ]
            fit_error = [
                record for record in records
                if record["view"] == view and record["actor"] in rule_fit_actors
                and record["error_type"] == error_class
            ]
            tune_normal = [
                record for record in records
                if record["view"] == view and record["actor"] in rule_tune_actors
                and record["error_type"] == NORMAL_CLASS
            ]
            tune_error = [
                record for record in records
                if record["view"] == view and record["actor"] in rule_tune_actors
                and record["error_type"] == error_class
            ]
            val_normal = [
                record for record in records
                if record["view"] == view and record["split"] == "val" and record["error_type"] == NORMAL_CLASS
            ]
            val_error = [
                record for record in records
                if record["view"] == view and record["split"] == "val" and record["error_type"] == error_class
            ]
            metrics = sorted(
                set.intersection(
                    {key for record in fit_normal for key in record["metrics"]},
                    {key for record in fit_error for key in record["metrics"]},
                )
            )
            candidates = []
            for metric in metrics:
                if "screen_separation_ratio" in metric:
                    continue
                normal_values = [record["metrics"][metric] for record in fit_normal if metric in record["metrics"]]
                error_values = [record["metrics"][metric] for record in fit_error if metric in record["metrics"]]
                best = _best_threshold(normal_values, error_values)
                if best is None or best["balanced_accuracy"] < MIN_TRAIN_BALANCED_ACCURACY:
                    continue
                tuning_score = _score_records(tune_normal, tune_error, metric, best)
                if tuning_score < MIN_TUNING_BALANCED_ACCURACY:
                    continue
                # Strict holdout: this score is attached only for reporting and
                # is never used to filter, rank, or select a runtime rule.
                validation_score = _score_records(val_normal, val_error, metric, best)
                candidates.append(
                    {
                        "metric": metric,
                        "operator": best["operator"],
                        "threshold": round(float(best["threshold"]), 6),
                        "train_balanced_accuracy": round(float(best["balanced_accuracy"]), 4),
                        "tuning_balanced_accuracy": round(float(tuning_score), 4),
                        "validation_balanced_accuracy": round(float(validation_score), 4),
                        "train_normal_n": len(normal_values),
                        "train_error_n": len(error_values),
                        "tuning_normal_n": len(tune_normal),
                        "tuning_error_n": len(tune_error),
                        "validation_normal_n": len(val_normal),
                        "validation_error_n": len(val_error),
                    }
                )
            candidates.sort(
                key=lambda rule: (rule["tuning_balanced_accuracy"], rule["train_balanced_accuracy"]),
                reverse=True,
            )
            selected = []
            used_families = set()
            for rule in candidates:
                family = _metric_family(rule["metric"])
                if family in used_families:
                    continue
                selected.append(rule)
                used_families.add(family)
                if len(selected) >= MAX_RULES_PER_ERROR_VIEW:
                    break
            error_rules[view][error_class] = selected
            best_text = ", ".join(
                f"{rule['metric']} {rule['operator']} {rule['threshold']:.3f} "
                f"(tune BA={rule['tuning_balanced_accuracy']:.2f}, "
                f"val BA={rule['validation_balanced_accuracy']:.2f})"
                for rule in selected
            ) or "유효한 단일 조건 없음"
            print(f"[{view}/{error_class}] {best_text}")

    recommended_views = {}
    responsible_views = {}
    for error_class in ERROR_CLASSES:
        ranking = []
        for view in VIEWS:
            rules = error_rules[view][error_class]
            best_rule = max(rules, key=lambda rule: rule["tuning_balanced_accuracy"], default=None)
            tuning_score = best_rule["tuning_balanced_accuracy"] if best_rule else 0.5
            # Report the strict-holdout performance of that same tune-selected
            # rule, not the maximum validation result from a different rule.
            validation_score = best_rule["validation_balanced_accuracy"] if best_rule else 0.5
            ranking.append(
                {
                    "view": view,
                    "best_tuning_balanced_accuracy": tuning_score,
                    "reported_validation_balanced_accuracy": validation_score,
                }
            )
        recommended_views[error_class] = sorted(
            ranking, key=lambda item: item["best_tuning_balanced_accuracy"], reverse=True
        )
        best_tuning = max(item["best_tuning_balanced_accuracy"] for item in ranking)
        responsible_views[error_class] = [
            item["view"]
            for item in ranking
            if item["best_tuning_balanced_accuracy"] >= max(0.75, best_tuning - 0.05)
        ]

    payload = {
        "schema_version": 2,
        "description": (
            "AI Hub 에어스쿼트의 2D camera keypoints와 3D ground truth에서 만든 시점별 조건. "
            "정상 범위는 train actor만 사용했다. 오류 조건은 train actor 내부의 분리된 fit/tune "
            "배우로 학습·선택하고, val actor balanced accuracy는 선택에 사용하지 않고 최종 보고만 "
            "했다. 이 조건은 전체 시퀀스 분류기의 증거/거부 "
            "규칙이며, 조건을 통과했다는 이유만으로 단독 정상 판정을 내리지 않는다."
        ),
        "dataset_path_at_generation": str(DATASET_PATH),
        "actor_split_path": str(SPLIT_PATH.relative_to(ROOT)),
        "camera_views": mapping.as_dict(),
        "camera_statistics": mapping.statistics,
        "camera_identification": {
            "sample_sequences": len(camera_sample),
            "method": "2D-vs-3D projection Procrustes + face ordering + weak-perspective camera direction",
        },
        "record_counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "rule_selection_actor_split": {
            "fit": sorted(rule_fit_actors),
            "tune": sorted(rule_tune_actors),
            "validation": sorted(actor for actor, split in actor_split.items() if split == "val"),
            "method": "actor-disjoint deterministic every-fifth-train-actor tuning split",
        },
        "normal_range_quantiles": {"warning": [0.10, 0.90], "hard": [0.025, 0.975]},
        "normal_ranges": normal_ranges,
        "error_rule_selection": {
            "objective": "balanced_accuracy(error vs normal)",
            "threshold_fit_split": "rule-fit actors (train subset)",
            "rule_selection_split": "rule-tune actors (disjoint train subset)",
            "reported_holdout_split": "val actors (not used for selection)",
            "min_train_balanced_accuracy": MIN_TRAIN_BALANCED_ACCURACY,
            "min_tuning_balanced_accuracy": MIN_TUNING_BALANCED_ACCURACY,
            "max_rules_per_error_view": MAX_RULES_PER_ERROR_VIEW,
        },
        "error_rules": error_rules,
        "recommended_views_by_error": recommended_views,
        "responsible_views_by_error": responsible_views,
        "runtime_decision_policy": {
            "normal": "sequence model says normal AND no repeated high-confidence condition violation",
            "error": "sequence model predicts the error AND matching condition evidence supports it",
            "uncertain": "model/condition disagreement, low pose quality, or insufficient rule support",
            "repeat_confirmation": "same high-confidence error in at least 2 of 3 repetitions per view",
            "session_fusion": "combine evidence by tuning-selected error responsibility; never majority-vote the three views",
        },
    }
    OUT_PATH.write_text(
        json.dumps(_finite_json(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
