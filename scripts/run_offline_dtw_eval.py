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
import argparse
import itertools
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.actor_split import load_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.dataset_config import DATASET_PATH  # noqa: E402
from ai_trainer.dtw_compare import PHASES, multi_reference_distance, resolve_weights  # noqa: E402
from ai_trainer.features import extract_all_features  # noqa: E402
from ai_trainer.heel_contact import estimate_sequence_heel_contact  # noqa: E402
from ai_trainer.lifting_dataset import load_actor_split  # noqa: E402
from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.phase_segmentation import segment_phases  # noqa: E402
from ai_trainer.reference_db_io import load_reference_db_by_level  # noqa: E402
from ai_trainer.reference_levels import DIFFICULTY_LEVELS, REFERENCE_CLASSES  # noqa: E402
from ai_trainer.reference_matching import (  # noqa: E402
    BINARY_DISTANCE_FEATURE_NAMES,
    INDETERMINATE_CLASS,
    UNSTABLE_CLASS,
    binary_distance_features,
    decide_reference_match,
)
from ai_trainer.reference_pipeline import build_operational_reference  # noqa: E402
from ai_trainer.scoring import (  # noqa: E402
    binary_roc_auc,
    distance_to_unit,
    fit_binary_logistic_calibration,
    fit_hybrid_calibration,
    fit_score_calibration,
)
from ai_trainer.skeleton_augmentation import augment_skeleton_sequence  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = ROOT / "configs" / "actor_split.json"
WEIGHTS_CFG_PATH = ROOT / "configs" / "dtw_feature_weights.json"
DB_DIR = ROOT / "output" / "reference_db"
OUT_DIR = ROOT / "output" / "dtw_eval"
CLASSES = list(REFERENCE_CLASSES)
REPORT_CLASSES = CLASSES + [INDETERMINATE_CLASS, UNSTABLE_CLASS]
EVAL_ALGORITHMS = ("dtw", "ddtw", "hybrid")


def build_augmented_query(query: dict, variant: dict, seed: int) -> dict:
    """Create one train-only skeleton variant and recompute its phase/features."""
    coords = augment_skeleton_sequence(
        query["coords"],
        duration_scale=float(variant.get("duration_scale", 1.0)),
        yaw_deg=float(variant.get("yaw_deg", 0.0)),
        jitter_std=float(variant.get("jitter_std", 0.0)),
        smoothing=float(variant.get("jitter_smoothing", 0.80)),
        seed=seed,
    )
    bounds = segment_phases(extract_phase_features(coords)).as_dict()
    meta = dict(query["meta"])
    meta["augmentation"] = str(variant.get("name", f"variant_{seed}"))
    meta["source_frame_count"] = int(query["coords"].shape[0])
    meta["augmented_frame_count"] = int(coords.shape[0])
    return {
        "coords": coords,
        "feat": extract_all_features(coords),
        "bounds": bounds,
        # Image-space heel evidence is not geometrically transformed.  It is
        # inherited only as a class-preserving auxiliary label for calibration.
        "heel_contact_2d": query.get("heel_contact_2d"),
        "heel_lift_max_delta": query.get("heel_lift_max_delta"),
        "meta": meta,
    }


def augment_calibration_queries(
    queries: list[dict],
    config: dict,
    *,
    enabled: bool,
) -> tuple[list[dict], dict]:
    """Augment selected train classes while retaining source actor identity."""
    excluded = set(config.get("excluded_classes", []))
    variants = list(config.get("variants", [])) if enabled else []
    base_seed = int(config.get("seed", 20260908))
    combined = list(queries)
    augmented_counts: Counter[str] = Counter()
    for query_index, query in enumerate(queries):
        class_label = query["meta"]["true_class"]
        if class_label in excluded:
            continue
        for variant_index, variant in enumerate(variants):
            seed = base_seed + query_index * 1009 + variant_index
            combined.append(build_augmented_query(query, variant, seed))
            augmented_counts[class_label] += 1
    return combined, {
        "enabled": bool(enabled and variants),
        "train_only": True,
        "validation_uses_original_only": True,
        "excluded_classes": sorted(excluded),
        "min_group_cv_objective_gain": float(
            config.get("min_group_cv_objective_gain", 0.01)
        ),
        "variants": variants,
        "original_counts_by_class": dict(
            Counter(query["meta"]["true_class"] for query in queries)
        ),
        "augmented_counts_by_class": dict(augmented_counts),
        "n_original": len(queries),
        "n_augmented": int(sum(augmented_counts.values())),
    }


def build_query(z: AiHubZip, seq, origin: str, model, device) -> dict | None:
    ref = build_operational_reference(z, seq, origin, model, device)
    if ref is None or ref.coords.shape[0] < 10:
        return None
    feat = extract_all_features(ref.coords)
    pf = extract_phase_features(ref.coords)
    bounds = segment_phases(pf).as_dict()
    heel_contact_2d, heel_lift_max_delta = (None, None)
    if ref.image_coords_2d is not None:
        heel_contact_2d, heel_lift_max_delta = estimate_sequence_heel_contact(
            ref.image_coords_2d,
            bounds.get("준비"),
        )
    return {
        "coords": ref.coords,
        "feat": feat,
        "bounds": bounds,
        "heel_contact_2d": heel_contact_2d,
        "heel_lift_max_delta": heel_lift_max_delta,
        "meta": {
            "actor": seq.actor,
            "level": seq.level,
            "true_class": seq.error_type,
            "rep": seq.rep,
            "origin": origin,
            "frame_range": list(ref.frame_range),
        },
    }


def classify(
    query: dict,
    db_tier: dict,
    weights_cfg: dict,
    weight_profile: str,
    top_k: int = 2,
    algorithm: str = "dtw",
) -> dict:
    per_class = {}
    for cls in CLASSES:
        medoids = db_tier[cls]
        w = resolve_weights(weights_cfg, weight_profile, class_label=cls)
        per_class[cls] = multi_reference_distance(
            query["feat"],
            query["bounds"],
            medoids,
            w,
            weights_cfg,
            top_k=top_k,
            algorithm=algorithm,
            class_label=cls,
        )
    return per_class


def classify_all_levels(
    query: dict,
    db_tier_by_level: dict,
    weights_cfg: dict,
    weight_profile: str,
    top_k: int = 2,
    algorithm: str = "dtw",
) -> dict:
    """한 query를 초급·중급·고급의 모든 자세 class와 비교한다."""
    return {
        level: classify(
            query,
            db_tier_by_level[level],
            weights_cfg,
            weight_profile,
            top_k=top_k,
            algorithm=algorithm,
        )
        for level in DIFFICULTY_LEVELS
    }


def blend_reference_result(
    dtw_result: dict,
    ddtw_result: dict,
    calibration: dict,
) -> dict:
    """Blend one level/class result after component-wise normalization."""
    alpha = float(calibration["alpha"])
    dtw_calib = calibration["components"]["dtw"]
    ddtw_calib = calibration["components"]["ddtw"]
    dtw_distance = float(distance_to_unit(dtw_result["min_distance"], dtw_calib))
    ddtw_distance = float(distance_to_unit(ddtw_result["min_distance"], ddtw_calib))
    dtw_topk = float(distance_to_unit(dtw_result["mean_topk_distance"], dtw_calib))
    ddtw_topk = float(distance_to_unit(ddtw_result["mean_topk_distance"], ddtw_calib))
    blended = dict(dtw_result)
    blended.update(
        {
            "algorithm": "hybrid",
            "min_distance": alpha * dtw_distance + (1.0 - alpha) * ddtw_distance,
            "mean_topk_distance": alpha * dtw_topk + (1.0 - alpha) * ddtw_topk,
            "component_distances": {"dtw": dtw_distance, "ddtw": ddtw_distance},
            "component_raw_distances": {
                "dtw": float(dtw_result["min_distance"]),
                "ddtw": float(ddtw_result["min_distance"]),
            },
        }
    )
    return blended


def classify_all_levels_hybrid(
    query: dict,
    db_tier_by_level: dict,
    weights_cfg: dict,
    weight_profile: str,
    calibration: dict,
    top_k: int = 2,
) -> dict:
    """Compare all levels/classes with normalized DTW+DDTW distances."""
    dtw_results = classify_all_levels(
        query,
        db_tier_by_level,
        weights_cfg,
        weight_profile,
        top_k=top_k,
        algorithm="dtw",
    )
    ddtw_results = classify_all_levels(
        query,
        db_tier_by_level,
        weights_cfg,
        weight_profile,
        top_k=top_k,
        algorithm="ddtw",
    )
    return {
        level: {
            class_label: blend_reference_result(
                dtw_results[level][class_label],
                ddtw_results[level][class_label],
                calibration,
            )
            for class_label in CLASSES
        }
        for level in DIFFICULTY_LEVELS
    }


def confusion_and_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    classes = REPORT_CLASSES
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
    target_metrics = [metrics[class_label] for class_label in CLASSES]
    supports = np.asarray([item["support"] for item in target_metrics], dtype=np.float64)
    total_support = max(float(supports.sum()), 1.0)
    macro = {
        name: float(np.mean([item[name] for item in target_metrics]))
        for name in ("precision", "recall", "f1")
    }
    weighted = {
        name: float(
            np.sum([item[name] * support for item, support in zip(target_metrics, supports)])
            / total_support
        )
        for name in ("precision", "recall", "f1")
    }
    return {
        "confusion_matrix": cm.tolist(),
        "classes": classes,
        "per_class": metrics,
        "accuracy": accuracy,
        "macro_average": macro,
        "weighted_average": weighted,
    }


def fit_rejection_calibration(results: list[dict], normal_threshold: float) -> dict:
    """Fit error acceptance, per-class robust scaling, and ambiguity margins."""
    error_classes = [class_label for class_label in CLASSES if class_label != "정상"]
    distances_by_true_class: dict[str, list[float]] = {
        class_label: [] for class_label in error_classes
    }
    raw_distances_by_result: list[tuple[str, dict[str, float]]] = []
    for result in results:
        true_class = result["meta"]["true_class"]
        if true_class not in distances_by_true_class:
            continue
        per_level = result["per_level"]
        by_class = {
            class_label: min(
                float(per_level[level][class_label]["min_distance"])
                for level in DIFFICULTY_LEVELS
            )
            for class_label in error_classes
        }
        true_distance = by_class[true_class]
        if np.isfinite(true_distance):
            distances_by_true_class[true_class].append(true_distance)
        raw_distances_by_result.append((true_class, by_class))

    fallback = float(normal_threshold) * 1.5
    thresholds = {}
    normalization = {}
    for class_label, distances in distances_by_true_class.items():
        finite = np.asarray(distances, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        thresholds[class_label] = (
            float(np.quantile(finite, 0.95) * 1.05) if finite.size else fallback
        )
        if finite.size:
            q25, median, q75 = np.percentile(finite, [25.0, 50.0, 75.0])
            iqr = max(float(q75 - q25), abs(float(median)) * 0.05, 1e-6)
        else:
            q25, median, q75, iqr = 0.0, fallback, fallback, max(fallback * 0.05, 1e-6)
        normalization[class_label] = {
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "iqr": float(iqr),
        }

    correct_relative_margins: list[float] = []
    correct_z_margins: list[float] = []
    for true_class, by_class in raw_distances_by_result:
        ordered = sorted(
            (distance, class_label)
            for class_label, distance in by_class.items()
            if np.isfinite(distance)
        )
        if len(ordered) >= 2 and ordered[0][1] == true_class:
            margin = (ordered[1][0] - ordered[0][0]) / max(abs(ordered[0][0]), 1e-8)
            correct_relative_margins.append(float(margin))
        z_ordered = sorted(
            (
                (distance - normalization[class_label]["median"])
                / normalization[class_label]["iqr"],
                class_label,
            )
            for class_label, distance in by_class.items()
            if np.isfinite(distance)
        )
        if len(z_ordered) >= 2 and z_ordered[0][1] == true_class:
            correct_z_margins.append(float(z_ordered[1][0] - z_ordered[0][0]))
    min_margin = (
        float(np.clip(np.quantile(correct_relative_margins, 0.10), 0.02, 0.25))
        if correct_relative_margins
        else 0.05
    )
    min_z_margin = (
        float(np.clip(np.quantile(correct_z_margins, 0.10), 0.02, 2.0))
        if correct_z_margins
        else 0.05
    )
    return {
        "error_distance_thresholds": thresholds,
        "error_distance_normalization": normalization,
        "min_error_relative_margin": min_margin,
        "min_error_z_margin": min_z_margin,
        "rejection_calibration_counts": {
            class_label: len(values)
            for class_label, values in distances_by_true_class.items()
        },
    }


def binary_normal_vs_error(y_true: list[str], y_pred: list[str]) -> float:
    correct = 0
    for t, p in zip(y_true, y_pred):
        t_bin = t == "정상"
        p_bin = p == "정상"
        correct += int(t_bin == p_bin)
    return correct / max(1, len(y_true))


def binary_classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    """Treat every non-normal output, including rejection, as error-positive."""
    truth = np.asarray([label != "정상" for label in y_true], dtype=bool)
    predicted = np.asarray([label != "정상" for label in y_pred], dtype=bool)
    tp = int(np.sum(truth & predicted))
    fp = int(np.sum(~truth & predicted))
    fn = int(np.sum(truth & ~predicted))
    tn = int(np.sum(~truth & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "positive_class": "비정상",
        "accuracy": (tp + tn) / max(1, len(truth)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def _auc_ready(scores: list[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    finite = values[np.isfinite(values)]
    floor = float(np.min(finite) - max(np.ptp(finite), 1.0)) if finite.size else -1.0
    return np.where(np.isfinite(values), values, floor)


def roc_auc_metrics(results: list[dict]) -> dict:
    """Calculate threshold-independent AUC from hierarchical distance scores."""
    y_true = [result["meta"]["true_class"] for result in results]
    class_scores: dict[str, list[float]] = {class_label: [] for class_label in CLASSES}
    binary_error_scores = []
    for result in results:
        decision = result["decision"]
        for class_label in CLASSES:
            distance = min(
                float(result["per_level"][level][class_label]["min_distance"])
                for level in DIFFICULTY_LEVELS
            )
            class_scores[class_label].append(-distance)
        if decision.normal_probability is not None:
            binary_error_scores.append(1.0 - float(decision.normal_probability))
        else:
            binary_error_scores.append(-class_scores["정상"][-1])

    by_class = {
        class_label: binary_roc_auc(
            np.asarray([truth == class_label for truth in y_true]),
            _auc_ready(scores),
        )
        for class_label, scores in class_scores.items()
    }
    finite_auc = [value for value in by_class.values() if value is not None]
    return {
        "score_definition": "class=-minimum_DTW_distance; binary=1-P(normal)",
        "binary_normal_vs_error": binary_roc_auc(
            np.asarray([truth != "정상" for truth in y_true]),
            _auc_ready(binary_error_scores),
        ),
        "one_vs_rest_by_class": by_class,
        "macro_one_vs_rest": float(np.mean(finite_auc)) if finite_auc else None,
    }


def fit_hierarchical_calibration_from_results(results: list[dict]) -> dict:
    """Fit both hierarchy stages from already-computed actor-disjoint DTW results."""
    normal_distances = []
    error_distances = []
    for result in results:
        distance = min(
            float(result["per_level"][level]["정상"]["min_distance"])
            for level in DIFFICULTY_LEVELS
        )
        target = (
            normal_distances
            if result["meta"]["true_class"] == "정상"
            else error_distances
        )
        target.append(distance)
    calibration = fit_score_calibration(
        np.asarray(normal_distances), np.asarray(error_distances)
    )
    calibration.update(
        {"n_normal": len(normal_distances), "n_error": len(error_distances)}
    )
    calibration.update(
        fit_rejection_calibration(
            results,
            float(calibration["normal_distance_threshold"]),
        )
    )
    calibration["binary_classifier"] = fit_binary_logistic_calibration(
        np.stack(
            [binary_distance_features(result["per_level"]) for result in results]
        ),
        np.asarray(
            [result["meta"]["true_class"] == "정상" for result in results],
            dtype=bool,
        ),
        BINARY_DISTANCE_FEATURE_NAMES,
    )
    return calibration


def actor_group_cross_validation(
    calibration_results: list[dict],
    rejection_config: dict,
    *,
    n_splits: int = 5,
) -> dict:
    """Evaluate calibration logic with actors kept wholly inside one fold."""
    actors = sorted({result["meta"]["actor"] for result in calibration_results})
    fold_count = min(max(2, int(n_splits)), len(actors))
    folds = [actors[index::fold_count] for index in range(fold_count)]
    out_of_fold = []
    fold_summaries = []
    for fold_index, validation_actors in enumerate(folds):
        held_out = set(validation_actors)
        train = [
            result
            for result in calibration_results
            if result["meta"]["actor"] not in held_out
        ]
        validation = [
            result
            for result in calibration_results
            if result["meta"]["actor"] in held_out
            and "augmentation" not in result["meta"]
        ]
        if not train or not validation:
            continue
        fold_calibration = fit_hierarchical_calibration_from_results(train)
        fold_results = []
        for result in validation:
            decision = decide_reference_match(
                result["per_level"],
                score_calibration=fold_calibration,
                rejection_config=rejection_config,
                heel_contact_2d=result.get("heel_contact_2d"),
            )
            fold_results.append({**result, "decision": decision})
        truth = [result["meta"]["true_class"] for result in fold_results]
        predicted = [result["decision"].predicted_class for result in fold_results]
        fold_metric = confusion_and_metrics(truth, predicted)
        fold_summaries.append(
            {
                "fold": fold_index,
                "validation_actors": validation_actors,
                "n": len(fold_results),
                "accuracy": fold_metric["accuracy"],
                "macro_f1": fold_metric["macro_average"]["f1"],
                "binary": binary_classification_metrics(truth, predicted),
            }
        )
        out_of_fold.extend(fold_results)

    truth = [result["meta"]["true_class"] for result in out_of_fold]
    predicted = [result["decision"].predicted_class for result in out_of_fold]
    metrics = confusion_and_metrics(truth, predicted)
    metrics["binary_normal_vs_error"] = binary_classification_metrics(
        truth, predicted
    )
    metrics["roc_auc"] = roc_auc_metrics(out_of_fold)
    return {
        "method": "5-fold actor-grouped out-of-fold calibration; augmentation in train folds only",
        "n_actors": len(actors),
        "n_sequences": len(out_of_fold),
        "folds": fold_summaries,
        "aggregate": metrics,
    }


def select_augmentation_by_group_cv(
    calibration_results: list[dict],
    rejection_config: dict,
    augmentation_summary: dict,
) -> tuple[list[dict], dict, dict]:
    """Select augmentation variants without consulting final validation actors.

    Every candidate is fitted on non-held-out actors (including their synthetic
    variants) and evaluated only on original sequences from held-out actors.
    """
    variant_names = [str(item["name"]) for item in augmentation_summary["variants"]]
    if not augmentation_summary["enabled"] or not variant_names:
        cv = actor_group_cross_validation(calibration_results, rejection_config)
        return calibration_results, cv, {
            "selection_metric": "mean(macro_f1, binary_f1)",
            "selected_variants": [],
            "candidates": [],
        }

    candidate_sets = [
        tuple(combo)
        for size in range(len(variant_names) + 1)
        for combo in itertools.combinations(variant_names, size)
    ]
    candidates = []
    best_key = None
    best_results = None
    best_cv = None
    best_names: tuple[str, ...] = ()
    baseline_results = None
    baseline_cv = None
    baseline_objective = None
    for selected_names in candidate_sets:
        selected = set(selected_names)
        candidate_results = [
            result
            for result in calibration_results
            if "augmentation" not in result["meta"]
            or result["meta"]["augmentation"] in selected
        ]
        cv = actor_group_cross_validation(candidate_results, rejection_config)
        aggregate = cv["aggregate"]
        macro_f1 = float(aggregate["macro_average"]["f1"])
        binary_f1 = float(aggregate["binary_normal_vs_error"]["f1"])
        objective = 0.5 * (macro_f1 + binary_f1)
        row = {
            "variants": list(selected_names),
            "n_train_rows": len(candidate_results),
            "objective": objective,
            "accuracy": float(aggregate["accuracy"]),
            "macro_f1": macro_f1,
            "binary_accuracy": float(
                aggregate["binary_normal_vs_error"]["accuracy"]
            ),
            "binary_f1": binary_f1,
            "binary_auc": aggregate["roc_auc"]["binary_normal_vs_error"],
        }
        candidates.append(row)
        if not selected_names:
            baseline_results = candidate_results
            baseline_cv = cv
            baseline_objective = objective
        # Prefer the stronger grouped-CV objective; on numerical ties keep the
        # simpler (fewer variants) calibration set.
        key = (objective, -len(selected_names))
        if best_key is None or key > best_key:
            best_key = key
            best_results = candidate_results
            best_cv = cv
            best_names = selected_names

    assert (
        best_results is not None
        and best_cv is not None
        and baseline_results is not None
        and baseline_cv is not None
        and baseline_objective is not None
    )
    minimum_gain = float(augmentation_summary["min_group_cv_objective_gain"])
    raw_best_names = best_names
    raw_best_objective = float(best_key[0])
    gain = raw_best_objective - float(baseline_objective)
    accepted = bool(best_names) and gain >= minimum_gain
    if not accepted:
        best_results = baseline_results
        best_cv = baseline_cv
        best_names = ()
    return best_results, best_cv, {
        "selection_metric": "mean(macro_f1, binary_f1)",
        "minimum_objective_gain": minimum_gain,
        "baseline_objective": float(baseline_objective),
        "raw_best_variants": list(raw_best_names),
        "raw_best_objective": raw_best_objective,
        "raw_best_gain": gain,
        "augmentation_accepted": accepted,
        "selected_variants": list(best_names),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate phase-aware weighted DTW or derivative DTW."
    )
    parser.add_argument(
        "--algorithm",
        choices=EVAL_ALGORITHMS,
        default="dtw",
        help="alignment algorithm (default: dtw; use hybrid for normalized DTW+DDTW)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional report path (default: output/dtw_eval/offline_eval_report.json)",
    )
    parser.add_argument(
        "--augmentation",
        choices=("targeted", "none"),
        default="targeted",
        help="train-fold skeleton augmentation (default: targeted; hip-error is excluded by config)",
    )
    args = parser.parse_args()
    algorithm = args.algorithm
    report_path = args.output if args.output is not None else OUT_DIR / "offline_eval_report.json"

    t_start = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TemporalLiftingNet(n_joints=18, hidden=128)
    model.load_state_dict(torch.load(ROOT / "output" / "lifting_baseline" / "model_best.pt", map_location=device))
    model.to(device).eval()

    weights_cfg = json.loads(WEIGHTS_CFG_PATH.read_text(encoding="utf-8"))
    weight_profile = weights_cfg.get("default_profile", "E_full_uniform")
    db = load_reference_db_by_level(DB_DIR)
    medoid_actor_ids = {
        reference["meta"]["actor_id"]
        for tier in db.values()
        for level_db in tier.values()
        for references in level_db.values()
        for reference in references
    }
    print(f"Reference DB medoid actor: {sorted(medoid_actor_ids)}")

    actor_to_split = load_actor_split(SPLIT_PATH)
    all_seqs = load_air_squat_sequences(DATASET_PATH)
    val_seqs = [os_ for os_ in all_seqs if actor_to_split.get(os_.seq.actor) == "val" and os_.seq.error_type in CLASSES]
    print(f"Validation query 대상: {len(val_seqs)}개 (actor {len({o.seq.actor for o in val_seqs})}명)")

    overlap = medoid_actor_ids & {o.seq.actor for o in val_seqs}
    assert not overlap, f"actor leakage 발견: {overlap}"
    print("actor-disjoint 확인 통과 (medoid actor와 val actor의 교집합 없음)")

    dataset = AiHubZip(DATASET_PATH)
    try:
        # ---- 1) validation query 구축 (한 번 계산해 평가에 재사용) ----
        print("\nValidation query 생성 중 (camera1 2D -> lifting model -> 정규화)...")
        t0 = time.time()
        queries = []
        for os_ in val_seqs:
            query = build_query(dataset, os_.seq, os_.origin, model, device)
            if query is not None:
                queries.append(query)
        if not queries:
            raise RuntimeError("유효한 validation query가 없습니다.")
        print(f"쿼리 {len(queries)}개 생성 완료 ({time.time()-t0:.1f}초)")

        # ---- 2) 세 난이도 정상 Reference 통합 matching-rate 보정 ----
        print("\n3난이도 정상 매칭률 calibration용 train 표본 생성 중...")
        t0 = time.time()
        medoid_keys = {
            (
                reference["meta"]["actor_id"],
                reference["meta"]["class_label"],
                reference["meta"]["repetition_id"],
            )
            for level_db in db["ground_truth"].values()
            for references in level_db.values()
            for reference in references
        }

        import random

        rng = random.Random(23)
        calibration_pool = defaultdict(list)
        for os_ in all_seqs:
            if actor_to_split.get(os_.seq.actor) != "train":
                continue
            key = (os_.seq.actor, os_.seq.error_type, os_.seq.rep)
            if key not in medoid_keys:
                calibration_pool[(os_.seq.level, os_.seq.error_type)].append(os_)

        calibration_sequences = []
        for level in DIFFICULTY_LEVELS:
            for class_label in CLASSES:
                pool = calibration_pool[(level, class_label)]
                rng.shuffle(pool)
                calibration_sequences.extend(pool[:8])

        normal_distances: list[float] = []
        error_distances: list[float] = []
        original_calibration_queries: list[dict] = []
        hybrid_component_distances = {
            "normal": {"dtw": [], "ddtw": []},
            "error": {"dtw": [], "ddtw": []},
        }
        normal_weights = resolve_weights(
            weights_cfg, weight_profile, class_label="정상"
        )
        for os_ in calibration_sequences:
            query = build_query(dataset, os_.seq, os_.origin, model, device)
            if query is None:
                continue
            original_calibration_queries.append(query)

        augmentation_config = weights_cfg.get("calibration_augmentation", {})
        calibration_queries, augmentation_summary = augment_calibration_queries(
            original_calibration_queries,
            augmentation_config,
            enabled=(
                args.augmentation == "targeted"
                and bool(augmentation_config.get("enabled", True))
            ),
        )
        print(
            "calibration 증강: "
            f"원본 {augmentation_summary['n_original']}개 + "
            f"증강 {augmentation_summary['n_augmented']}개 "
            f"(제외: {', '.join(augmentation_summary['excluded_classes']) or '없음'})"
        )

        for query in calibration_queries:
            if algorithm == "hybrid":
                component_best: dict[str, float] = {}
                for component in ("dtw", "ddtw"):
                    component_distances = []
                    for level in DIFFICULTY_LEVELS:
                        result = multi_reference_distance(
                            query["feat"],
                            query["bounds"],
                            db["operational"][level]["정상"],
                            normal_weights,
                            weights_cfg,
                            top_k=2,
                            algorithm=component,
                            class_label="정상",
                        )
                        component_distances.append(result["min_distance"])
                    component_best[component] = min(component_distances)
                target_name = (
                    "normal"
                    if query["meta"]["true_class"] == "정상"
                    else "error"
                )
                for component in ("dtw", "ddtw"):
                    hybrid_component_distances[target_name][component].append(
                        component_best[component]
                    )
                continue
            distances = []
            for level in DIFFICULTY_LEVELS:
                result = multi_reference_distance(
                    query["feat"],
                    query["bounds"],
                    db["operational"][level]["정상"],
                    normal_weights,
                    weights_cfg,
                    top_k=2,
                    algorithm=algorithm,
                    class_label="정상",
                )
                distances.append(result["min_distance"])
            best_distance = min(distances)
            target = (
                normal_distances
                if query["meta"]["true_class"] == "정상"
                else error_distances
            )
            target.append(best_distance)

        if algorithm == "hybrid":
            calibration = fit_hybrid_calibration(
                np.asarray(hybrid_component_distances["normal"]["dtw"]),
                np.asarray(hybrid_component_distances["normal"]["ddtw"]),
                np.asarray(hybrid_component_distances["error"]["dtw"]),
                np.asarray(hybrid_component_distances["error"]["ddtw"]),
            )
            n_normal = len(hybrid_component_distances["normal"]["dtw"])
            n_error = len(hybrid_component_distances["error"]["dtw"])
        else:
            calibration = fit_score_calibration(
                np.asarray(normal_distances), np.asarray(error_distances)
            )
            n_normal = len(normal_distances)
            n_error = len(error_distances)
        calibration.update(
            {"n_normal": n_normal, "n_error": n_error}
        )
        rejection_results = []
        for query in calibration_queries:
            if algorithm == "hybrid":
                per_level = classify_all_levels_hybrid(
                    query,
                    db["operational"],
                    weights_cfg,
                    weight_profile,
                    calibration,
                    top_k=2,
                )
            else:
                per_level = classify_all_levels(
                    query,
                    db["operational"],
                    weights_cfg,
                    weight_profile,
                    top_k=2,
                    algorithm=algorithm,
                )
            rejection_results.append(
                {
                    "meta": query["meta"],
                    "per_level": per_level,
                    "heel_contact_2d": query.get("heel_contact_2d"),
                }
            )
        selected_calibration_results, calibration_group_cv, augmentation_selection = (
            select_augmentation_by_group_cv(
                rejection_results,
                weights_cfg.get("rejection", {}),
                augmentation_summary,
            )
        )
        hierarchical_calibration = fit_hierarchical_calibration_from_results(
            selected_calibration_results
        )
        calibration.update(hierarchical_calibration)
        augmentation_summary["selection"] = augmentation_selection
        augmentation_summary["n_selected_train_rows"] = len(
            selected_calibration_results
        )
        selected_counts = Counter(
            result["meta"]["true_class"] for result in selected_calibration_results
        )
        augmentation_summary["selected_counts_by_class"] = dict(selected_counts)
        print(
            "group-CV 선택 증강: "
            f"{augmentation_selection['selected_variants'] or ['없음']}"
        )
        if algorithm == "hybrid":
            print(f"hybrid alpha={calibration['alpha']:.2f}")
        print(
            f"calibration 완료 ({time.time()-t0:.1f}초): "
            f"distance_threshold={calibration['normal_distance_threshold']:.3f}, "
            f"match_threshold={calibration['normal_match_threshold']:.1f}% "
            f"(n_normal={n_normal}, n_error={n_error})"
        )

        # ---- 3) 실제 live와 같은 세 난이도 any-normal-match 판정 평가 ----
        print("\nOperational 3난이도 통합 판정 평가 중...")
        t0 = time.time()
        results = []
        for query in queries:
            if algorithm == "hybrid":
                per_level = classify_all_levels_hybrid(
                    query,
                    db["operational"],
                    weights_cfg,
                    weight_profile,
                    calibration,
                    top_k=2,
                )
            else:
                per_level = classify_all_levels(
                    query,
                    db["operational"],
                    weights_cfg,
                    weight_profile,
                    top_k=2,
                    algorithm=algorithm,
                )
            decision = decide_reference_match(
                per_level,
                score_calibration=calibration,
                rejection_config=weights_cfg.get("rejection"),
                heel_contact_2d=query.get("heel_contact_2d"),
            )
            results.append(
                {
                    "meta": query["meta"],
                    "per_level": per_level,
                    "decision": decision,
                    "heel_contact_2d": query.get("heel_contact_2d"),
                    "heel_lift_max_delta": query.get("heel_lift_max_delta"),
                }
            )
        print(f"평가 완료 ({time.time()-t0:.1f}초)")

        y_true = [result["meta"]["true_class"] for result in results]
        y_pred = [result["decision"].predicted_class for result in results]
        metrics = confusion_and_metrics(y_true, y_pred)
        metrics["binary_normal_vs_error_acc"] = binary_normal_vs_error(
            y_true, y_pred
        )
        metrics["binary_normal_vs_error"] = binary_classification_metrics(
            y_true, y_pred
        )
        metrics["roc_auc"] = roc_auc_metrics(results)
        metrics["heel_evidence_counts"] = dict(
            Counter(
                "stable"
                if result.get("heel_contact_2d") is True
                else "lifted"
                if result.get("heel_contact_2d") is False
                else "unknown"
                for result in results
            )
        )
        by_level = defaultdict(lambda: [0, 0])
        for result, true_class, predicted_class in zip(results, y_true, y_pred):
            level = result["meta"]["level"]
            by_level[level][1] += 1
            by_level[level][0] += int(true_class == predicted_class)
        metrics["accuracy_by_actor_level"] = {
            level: correct / count
            for level, (correct, count) in by_level.items()
        }
        print(
            f"accuracy={metrics['accuracy']:.3f}, "
            f"macro_f1={metrics['macro_average']['f1']:.3f}, "
            f"binary(정상vs오류)={metrics['binary_normal_vs_error_acc']:.3f}, "
            f"binary_f1={metrics['binary_normal_vs_error']['f1']:.3f}, "
            f"binary_auc={metrics['roc_auc']['binary_normal_vs_error']:.3f}"
        )

        report: dict = {
            "schema_version": 4,
            "algorithm": algorithm,
            "decision_rule": (
                "초급·중급·고급의 정상/오류 DTW 거리 meta-feature로 1단계 정상 확률을 "
                "계산하고, 비정상이면 class별 median/IQR 정규화 거리로 2단계 오류를 선택. "
                "오류별 거리와 z-margin 및 2D heel 근거를 통과할 때만 오류 class로 확정"
            ),
            "levels": list(DIFFICULTY_LEVELS),
            "classes": CLASSES,
            "weight_profile": weight_profile,
            "dtw_alignment": weights_cfg.get("dtw_alignment", {}),
            "rejection_defaults": weights_cfg.get("rejection", {}),
            "hierarchical_matching": {
                "stage_1": "regularized logistic P(normal) from 7 DTW distance meta-features",
                "stage_2": "per-error median/IQR normalized distance with z-margin rejection",
                "class_phase_weights": weights_cfg.get("class_phase_weights", {}),
            },
            "heel_evidence": {
                "enabled": True,
                "offline_proxy": "AIHub 준비 phase를 실시간 카운트다운 baseline 대신 사용",
                "threshold_torso_ratio": 0.10,
                "consecutive_frames": 2,
            },
            "calibration_augmentation": augmentation_summary,
            "n_queries": len(queries),
            "score_calibration": calibration,
            "calibration_actor_group_cv": calibration_group_cv,
            "operational": {"any_level_normal_match": metrics},
        }

        print("\n예시 Validation 쿼리:")
        examples = []
        rng2 = np.random.default_rng(5)
        sample_indices = rng2.choice(
            len(results), size=min(8, len(results)), replace=False
        )
        for index in sample_indices:
            result = results[int(index)]
            decision = result["decision"]
            selected = decision.selected_result
            top_features = sorted(
                selected["best_detail"]["per_feature_contrib"].items(),
                key=lambda item: -item[1],
            )[:3]
            example = {
                "actor": result["meta"]["actor"],
                "actor_level": result["meta"]["level"],
                "true_class": result["meta"]["true_class"],
                "predicted_class": decision.predicted_class,
                "matched_reference_level": decision.matched_level,
                "normal_match_rate": decision.match_rate,
                "normal_match_by_level": decision.normal_match_by_level,
                "distance_by_class": decision.raw_distance_by_class,
                "top_contributing_features": top_features,
                "heel_contact_2d": result.get("heel_contact_2d"),
                "heel_lift_max_delta": result.get("heel_lift_max_delta"),
            }
            examples.append(example)
            print(
                f"  {example['actor']}({example['actor_level']}) "
                f"true={example['true_class']:8s} pred={example['predicted_class']:8s} "
                f"matched={example['matched_reference_level']} "
                f"match={example['normal_match_rate']:5.1f}%"
            )
        report["examples"] = examples

        report["elapsed_sec"] = time.time() - t_start
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"report_path={report_path}")
        print(
            f"\n전체 완료 ({report['elapsed_sec']:.1f}초). "
            f"저장: {OUT_DIR / 'offline_eval_report.json'}"
        )
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
