#!/usr/bin/env python3
"""Audit-only actor-disjoint evaluation of the current production classifier.

This script never mutates ``output/reference_db``.  It reconstructs fold-local
operational medoids in memory from the same 160 deterministic source candidates
used by the reference builder, then evaluates every usable source sequence with
its actor held out.
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from ai_trainer.actor_split import load_all_air_squat_sequences
from ai_trainer.aihub_zip import AiHubZip
from ai_trainer.clustering import kmedoids, pairwise_dtw_distance_matrix, sequence_feature_matrix
from ai_trainer.common_skeleton import to_common_skeleton
from ai_trainer.dtw_compare import multi_reference_distance, resolve_weights
from ai_trainer.features import extract_all_features
from ai_trainer.heel_semantic import heel_motion_evidence, validate_heel_candidate
from ai_trainer.lifting_model import TemporalLiftingNet
from ai_trainer.online_dtw import validate_production_candidate
from ai_trainer.phase_features import extract_phase_features
from ai_trainer.phase_segmentation import segment_phases
from ai_trainer.reference_pipeline import build_ground_truth_reference, build_operational_reference
from ai_trainer.two_d_diagnostic import extract_2d_features
from scripts.build_reference_db import (
    CLASSES, DTW_RADIUS, K_MEDOIDS, N_CANDIDATES_PER_CLASS, SEED, TL_ZIP, VL_ZIP,
    sample_candidates,
)

UNKNOWN = "자세추정불확실"


def source_id(item) -> str:
    s = item.seq
    return f"{item.origin}:{s.error_type}:{s.level}:{s.actor}:rep{s.rep}"


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else ["source_id", "reason"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    per_class = {}
    total = len(y_true)
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    for label in CLASSES:
        tp = sum(t == label and p == label for t, p in zip(y_true, y_pred))
        fp = sum(t != label and p == label for t, p in zip(y_true, y_pred))
        fn = sum(t == label and p != label for t, p in zip(y_true, y_pred))
        tn = total - tp - fp - fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": sum(t == label for t in y_true), "tp": tp, "fp": fp,
            "tn": tn, "fn": fn, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1,
        }
    support = np.asarray([per_class[c]["support"] for c in CLASSES], dtype=float)
    def average(key: str, weighted=False) -> float:
        values = np.asarray([per_class[c][key] for c in CLASSES], dtype=float)
        return float(np.average(values, weights=support)) if weighted else float(values.mean())
    unknown_n = sum(p == UNKNOWN for p in y_pred)
    return {
        "total": total, "correct": correct, "incorrect": total - correct,
        "accuracy": correct / total if total else 0.0,
        "balanced_accuracy": average("recall"),
        "macro_precision": average("precision"), "macro_recall": average("recall"),
        "macro_f1": average("f1"), "weighted_precision": average("precision", True),
        "weighted_recall": average("recall", True), "weighted_f1": average("f1", True),
        "unknown_count": unknown_n, "unknown_rate": unknown_n / total if total else 0.0,
        "coverage": (total - unknown_n) / total if total else 0.0,
        "per_class": per_class,
    }


def auc_binary(target: np.ndarray, score: np.ndarray) -> float | None:
    target = np.asarray(target, dtype=bool); score = np.asarray(score, dtype=float)
    pos, neg = score[target], score[~target]
    if not len(pos) or not len(neg):
        return None
    # Pairwise definition handles ties exactly and is inexpensive at N=404.
    return float(((pos[:, None] > neg[None, :]).sum()
                  + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def confusion_rows(y_true, y_pred):
    outcomes = [*CLASSES, UNKNOWN]
    matrix = np.zeros((len(CLASSES), len(outcomes)), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[CLASSES.index(truth), outcomes.index(pred)] += 1
    rows, normalized = [], []
    for i, label in enumerate(CLASSES):
        rows.append({"actual": label, **{outcomes[j]: int(matrix[i, j]) for j in range(len(outcomes))}})
        denom = max(int(matrix[i].sum()), 1)
        normalized.append({"actual": label, **{outcomes[j]: float(matrix[i, j] / denom) for j in range(len(outcomes))}})
    return matrix, rows, normalized


def build_sample(item, zips, model, device) -> tuple[dict | None, str | None]:
    try:
        gt = build_ground_truth_reference(zips[item.origin], item.seq, item.origin)
        op = build_operational_reference(zips[item.origin], item.seq, item.origin, model, device)
        if gt is None or op is None:
            return None, "invalid sequence"
        if len(gt.coords) < 10 or len(op.coords) < 10:
            return None, "insufficient frames"
        _, points = zips[item.origin].read_2d(item.seq, 1)
        start, end = op.frame_range
        raw2d = to_common_skeleton(points[start:end + 1])
        if len(raw2d) < 10 or not np.isfinite(raw2d).all() or not np.isfinite(op.coords).all():
            return None, "invalid sequence"
        gt_phase = extract_phase_features(gt.coords)
        gt_bounds = segment_phases(gt_phase).as_dict()
        op_bounds = segment_phases(extract_phase_features(op.coords)).as_dict()
        _, two_d = extract_2d_features(raw2d)
        return {
            "id": source_id(item), "origin": item.origin, "actor": item.seq.actor,
            "label": item.seq.error_type, "level": item.seq.level, "rep": item.seq.rep,
            "gt_dtw": sequence_feature_matrix(
                gt_phase.pelvis_height, gt_phase.knee_flexion_deg, gt_phase.hip_flexion_deg
            ),
            "op_coords": op.coords, "op_feat": extract_all_features(op.coords),
            "gt_bounds": gt_bounds, "op_bounds": op_bounds, "raw2d": raw2d,
            "two_d": two_d,
        }, None
    except FileNotFoundError as error:
        reason = "missing annotation" if "annotation" in str(error).lower() else "missing source data"
        return None, reason
    except Exception as error:
        return None, f"feature extraction failure: {type(error).__name__}: {error}"


def main() -> int:
    started = time.time()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = ROOT / "output" / "diagnostics" / "actor_disjoint_validation" / stamp
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "configs" / "dtw_feature_weights.json").read_text(encoding="utf-8"))
    device = torch.device("cpu")
    model = TemporalLiftingNet(n_joints=18, hidden=128)
    model.load_state_dict(torch.load(
        ROOT / "output" / "lifting_baseline" / "model_best.pt", map_location=device,
        weights_only=True,
    ))
    model.eval()
    all_items = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    actors = sorted({x.seq.actor for x in all_items})
    candidate_ids = {
        source_id(x)
        for label in CLASSES
        for x in sample_candidates(all_items, label, N_CANDIDATES_PER_CLASS, SEED)
    }
    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}
    samples, excluded = [], []
    try:
        for index, item in enumerate(all_items, 1):
            sample, reason = build_sample(item, zips, model, device)
            if sample is None:
                excluded.append({"source_id": source_id(item), "actor": item.seq.actor,
                                 "class": item.seq.error_type, "reason": reason})
            else:
                sample["candidate"] = sample["id"] in candidate_ids
                samples.append(sample)
            if index % 20 == 0 or index == len(all_items):
                print(f"[BUILD] {index}/{len(all_items)} pass={len(samples)} excluded={len(excluded)}", flush=True)
    finally:
        for archive in zips.values():
            archive.close()

    # Precompute candidate GT distances once. A fold only masks its actor.
    candidates = [s for s in samples if s["candidate"]]
    gt_pair = {}
    for label in CLASSES:
        group = [s for s in candidates if s["label"] == label]
        matrix = pairwise_dtw_distance_matrix([s["gt_dtw"] for s in group], radius=DTW_RADIUS)
        for i, left in enumerate(group):
            for j, right in enumerate(group):
                gt_pair[left["id"], right["id"]] = float(matrix[i, j])

    predictions = []
    actor_leakage = source_leakage = self_reference = 0
    for fold_index, actor in enumerate(actors, 1):
        fold_refs = {}
        reference_actor_set = set()
        reference_source_set = set()
        for label in CLASSES:
            pool = [s for s in candidates if s["label"] == label and s["actor"] != actor]
            distance = np.asarray([[gt_pair[a["id"], b["id"]] for b in pool] for a in pool])
            medoid_indices, assignment = kmedoids(distance, k=min(K_MEDOIDS, len(pool)), seed=SEED)
            refs = []
            for rank, idx in enumerate(medoid_indices):
                selected = pool[int(idx)]
                refs.append({
                    "feat": selected["op_feat"], "bounds": selected["gt_bounds"],
                    "meta": {"medoid_id": selected["id"], "actor_id": selected["actor"],
                             "class_label": label, "cluster_size": int((assignment == rank).sum())},
                })
                reference_actor_set.add(selected["actor"]); reference_source_set.add(selected["id"])
            fold_refs[label] = refs
        validation = [s for s in samples if s["actor"] == actor]
        actor_leakage += int(bool(reference_actor_set & {actor}))
        for query in validation:
            source_leakage += int(query["id"] in reference_source_set)
            self_reference += int(any(
                r["meta"]["medoid_id"] == query["id"] for refs in fold_refs.values() for r in refs
            ))
            per_class = {}
            for label in CLASSES:
                weights = resolve_weights(config, config["default_profile"], class_label=label)
                per_class[label] = multi_reference_distance(
                    query["op_feat"], query["op_bounds"], fold_refs[label], weights, config, top_k=2
                )
            distances = {label: float(per_class[label]["min_distance"]) for label in CLASSES}
            ordered = sorted(CLASSES, key=distances.get); raw = ordered[0]
            sem = config["2d_semantic_validation"]
            depth_pass = (
                query["two_d"]["knee_excursion"] >= float(sem["knee_excursion_min_deg"])
                and query["two_d"]["hip_excursion"] >= float(sem["hip_excursion_min_deg"])
            )
            final = validate_production_candidate(raw, depth_2d_pass=depth_pass, consistency_pass=True)
            heel_cfg = config["heel_semantic_validation"]
            heel = heel_motion_evidence(
                query["raw2d"], no_max=float(heel_cfg["no_evidence_max"]),
                yes_min=float(heel_cfg["positive_evidence_min"]),
            )
            relative_margin = (distances[ordered[1]] - distances[ordered[0]]) / max(distances[ordered[0]], 1e-8)
            final, reason = validate_heel_candidate(raw, final, heel, relative_margin)
            predictions.append({
                "fold_actor": actor, "source_id": query["id"], "origin_zip": query["origin"],
                "actor": query["actor"], "true_class": query["label"], "raw_class": raw,
                "final_class": final, "correct": final == query["label"],
                "depth_2d_pass": depth_pass, "knee_excursion_2d": query["two_d"]["knee_excursion"],
                "hip_excursion_2d": query["two_d"]["hip_excursion"],
                "heel_verdict": heel["verdict"], "heel_value": heel["value"],
                "semantic_reason": reason or ("UNKNOWN_DEPTH_CONFLICT" if raw == CLASSES[2] and depth_pass
                    else "UNKNOWN_DEPTH_AMBIGUOUS" if raw == CLASSES[0] and not depth_pass else "VALIDATED"),
                "best_class": ordered[0], "second_class": ordered[1],
                "margin": distances[ordered[1]] - distances[ordered[0]],
                "relative_margin": relative_margin,
                **{f"distance_{label}": distances[label] for label in CLASSES},
                "reference_ids": "|".join(
                    r["meta"]["medoid_id"] for refs in fold_refs.values() for r in refs
                ),
            })
        print(f"[FOLD] {fold_index}/{len(actors)} actor={actor} n={len(validation)}", flush=True)

    y_true = [r["true_class"] for r in predictions]
    y_pred = [r["final_class"] for r in predictions]
    overall = metrics(y_true, y_pred)
    classified = [r for r in predictions if r["final_class"] != UNKNOWN]
    classified_metrics = metrics(
        [r["true_class"] for r in classified], [r["final_class"] for r in classified]
    )
    matrix, matrix_rows, normalized_rows = confusion_rows(y_true, y_pred)
    roc = {}
    for label in CLASSES:
        roc[label] = auc_binary(
            np.asarray([t == label for t in y_true]),
            -np.asarray([r[f"distance_{label}"] for r in predictions]),
        )
    supports = np.asarray([sum(t == label for t in y_true) for label in CLASSES], dtype=float)
    roc["macro"] = float(np.mean([roc[c] for c in CLASSES]))
    roc["weighted"] = float(np.average([roc[c] for c in CLASSES], weights=supports))
    flat_target = np.asarray([[t == c for c in CLASSES] for t in y_true]).ravel()
    flat_score = -np.asarray([[r[f"distance_{c}"] for c in CLASSES] for r in predictions]).ravel()
    roc["micro"] = auc_binary(flat_target, flat_score)

    actor_rows = []
    for actor in actors:
        selected = [r for r in predictions if r["actor"] == actor]
        actor_rows.append({
            "actor": actor, "sample_count": len(selected),
            "accuracy": sum(r["correct"] for r in selected) / len(selected),
            "unknown_rate": sum(r["final_class"] == UNKNOWN for r in selected) / len(selected),
            "correct": sum(r["correct"] for r in selected),
            "incorrect": sum(not r["correct"] for r in selected),
        })

    rng = random.Random(20260908); boot_accuracy, boot_f1 = [], []
    by_actor = {a: [r for r in predictions if r["actor"] == a] for a in actors}
    for _ in range(2000):
        draw = [rng.choice(actors) for _ in actors]
        rows = [r for actor in draw for r in by_actor[actor]]
        result = metrics([r["true_class"] for r in rows], [r["final_class"] for r in rows])
        boot_accuracy.append(result["accuracy"]); boot_f1.append(result["macro_f1"])
    confidence = {
        "unit": "actor", "replicates": 2000,
        "accuracy_95_ci": [float(x) for x in np.percentile(boot_accuracy, [2.5, 97.5])],
        "macro_f1_95_ci": [float(x) for x in np.percentile(boot_f1, [2.5, 97.5])],
    }

    split = {
        "protocol": "Leave-One-Actor-Out", "actors": actors, "fold_count": len(actors),
        "folds": [{"actor": a, "validation_source_ids": [r["source_id"] for r in predictions if r["actor"] == a]}
                  for a in actors],
    }
    summary = {
        "name": "Production Classifier Actor-Disjoint Validation v1",
        "protocol": "Leave-One-Actor-Out", "production_weight_profile": config["default_profile"],
        "total_source_sequences": len(all_items), "evaluated_sequences": len(predictions),
        "excluded_sequences": len(excluded), "exclusion_reasons": dict(Counter(r["reason"] for r in excluded)),
        "actors": len(actors), "folds": len(actors), "class_distribution": dict(Counter(y_true)),
        "actor_leakage_count": actor_leakage, "source_leakage_count": source_leakage,
        "validation_self_reference_count": self_reference,
        "overall": overall, "classified_only": classified_metrics, "roc_auc": roc,
        "confidence_interval": confidence, "confusion_matrix": matrix.tolist(),
        "labels": CLASSES, "outcomes": [*CLASSES, UNKNOWN],
        "evaluation_differences": [
            "Offline source sequences use camera1 annotated segments; live tracking/REP boundary quality gates are not simulated.",
            "2D depth baseline uses the first five annotated frames, not interactive standing calibration.",
            "Patch 2A makes 2D/3D consistency diagnostic-only; therefore it is not a hard validation input.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "actor_split.json").write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "roc_auc.json").write_text(json.dumps(roc, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out / "predictions.csv", predictions)
    write_csv(out / "excluded_samples.csv", excluded)
    write_csv(out / "actor_metrics.csv", actor_rows)
    write_csv(out / "per_class_metrics.csv", [
        {"class": label, **overall["per_class"][label], "roc_auc": roc[label]} for label in CLASSES
    ])
    write_csv(out / "confusion_matrix.csv", matrix_rows)
    write_csv(out / "confusion_matrix_normalized.csv", normalized_rows)
    readme = f"""Production Classifier Actor-Disjoint Validation v1

Protocol: Leave-One-Actor-Out across {len(actors)} recovered actor IDs.
Evaluated: {len(predictions)} / {len(all_items)} source sequences.
Reference construction: deterministic 160-candidate pool, held-out actor removed,
fold-local k-medoids selected in memory, operational lifting coordinates used.
Production policy: {config['default_profile']} Final DTW, active 2D depth semantic,
active heel semantic, UNKNOWN preserved. Patch 2B candidate policy is not used.

ROC-AUC uses score(class) = -DTW_distance(class). It evaluates continuous DTW
ranking and is not the hard-label performance after semantic/UNKNOWN handling.

Limitations:
- This is actor-disjoint AI Hub offline validation, not live-webcam accuracy.
- Camera1 annotated clips replace interactive tracking and REP segmentation.
- 2D baseline uses the first five annotated frames.
"""
    (out / "README.txt").write_text(readme, encoding="utf-8")
    print(f"[DONE] {out}", flush=True)
    print(json.dumps({"overall": overall, "roc_auc": roc, "ci": confidence}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
