"""Quality-PASS source offline shadow evaluation of heel-error guard policies."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trainer.actor_split import load_all_air_squat_sequences
from ai_trainer.aihub_zip import AiHubZip
from ai_trainer.common_skeleton import to_common_skeleton
from ai_trainer.dtw_compare import phase_aware_weighted_dtw, resolve_weights
from ai_trainer.features import extract_all_features
from ai_trainer.heel_policy_shadow import (
    HEEL, NORMAL, apply_policy, classification_metrics, empirical_evidence,
    evidence_table, relative_heel_margin,
)
from ai_trainer.heel_semantic import heel_motion_evidence
from ai_trainer.offline_pose_benchmark import OfflinePoseBenchmark, robust_location_scale
from ai_trainer.phase_features import extract_phase_features
from ai_trainer.phase_segmentation import segment_phases
from ai_trainer.reference_pipeline import build_ground_truth_reference
from ai_trainer.reference_quality_audit import audit_sequence
from scripts.build_reference_db import TL_ZIP, VL_ZIP, N_CANDIDATES_PER_CLASS, SEED, sample_candidates

POLICIES = ("A_RAW_DTW", "B_NO_REJECT", "C_YES_REQUIRED", "D_HEEL_DIRECT_2CLASS", "E_PRODUCTION")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def file_state(path: Path) -> dict:
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns}


def pairwise_dtw(samples: list[dict], cfg: dict, weights: dict) -> tuple[list[dict], np.ndarray]:
    n = len(samples); distances = np.full((n, n), np.inf, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = phase_aware_weighted_dtw(samples[i]["features"], samples[i]["bounds"],
                                             samples[j]["features"], samples[j]["bounds"], weights, cfg)["total"]
            distances[i, j] = distances[j, i] = value
    rows = []
    for i, sample in enumerate(samples):
        normal_distance = min(distances[i, j] for j, ref in enumerate(samples) if j != i and ref["truth"] == NORMAL)
        heel_distance = min(distances[i, j] for j, ref in enumerate(samples) if j != i and ref["truth"] == HEEL)
        raw = HEEL if heel_distance < normal_distance else NORMAL
        rows.append({"sample_id": sample["sample_id"], "ground_truth": sample["truth"],
                     "raw_prediction": raw, "normal_distance": float(normal_distance),
                     "heel_distance": float(heel_distance),
                     "relative_margin": relative_heel_margin(normal_distance, heel_distance),
                     "heel_evidence": sample["evidence"], "heel_score": sample["heel_score"]})
    return rows, distances


def heel_direct_predictions(samples: list[dict]) -> list[str]:
    vectors = np.stack([s["heel_vector"] for s in samples])
    output = []
    for i, sample in enumerate(samples):
        train = np.delete(vectors, i, axis=0)
        _, scale = robust_location_scale(train)
        by_label = {}
        for label in (NORMAL, HEEL):
            candidates = [j for j, ref in enumerate(samples) if j != i and ref["truth"] == label]
            by_label[label] = min(float(np.linalg.norm((vectors[i] - vectors[j]) / scale)) for j in candidates)
        output.append(min(by_label, key=by_label.get))
        sample["heel_only_normal_distance"] = by_label[NORMAL]
        sample["heel_only_heel_distance"] = by_label[HEEL]
        sample["heel_only_margin"] = relative_heel_margin(by_label[NORMAL], by_label[HEEL])
    return output


def preference_band(margin: float, weak_cut: float, strong_cut: float) -> str:
    if margin >= strong_cut: return "Strong Heel"
    if margin > 0: return "Weak Heel"
    if margin >= -weak_cut: return "Tie/Ambiguous"
    return "Normal preference"


def evaluate_rows(base_rows: list[dict], direct: list[str]) -> tuple[list[dict], list[dict]]:
    expanded, metrics = [], []
    for policy in POLICIES:
        evaluated = []
        for row, direct_prediction in zip(base_rows, direct):
            result, reason = apply_policy(policy, row["raw_prediction"], row["heel_evidence"], direct_prediction)
            evaluated.append({"truth": row["ground_truth"], "result": result})
            expanded.append({**row, "policy": policy, "heel_direct_prediction": direct_prediction,
                             "result": result, "reason": reason or ""})
        metrics.append({"policy": policy, **classification_metrics(evaluated)})
    return expanded, metrics


def main() -> int:
    started = time.perf_counter()
    out = ROOT / "output/diagnostics/heel_policy_shadow_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    db_paths = (ROOT / "output/reference_db/manifest.json", ROOT / "output/reference_db/sequences.npz")
    before = {str(p): file_state(p) for p in db_paths}
    cfg = json.loads((ROOT / "configs/dtw_feature_weights.json").read_text(encoding="utf-8"))
    bc, hs = cfg["baseline_calibration"], cfg["heel_semantic_validation"]
    weights = resolve_weights(cfg, cfg["default_profile"], None)
    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}
    all_sequences = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    samples = []
    for label in (NORMAL, HEEL):
        for obj in sample_candidates(all_sequences, label, N_CANDIDATES_PER_CLASS, seed=SEED):
            ref = build_ground_truth_reference(zips[obj.origin], obj.seq, obj.origin)
            if ref is None: continue
            qa = audit_sequence(ref.coords, hip_standing_min=bc["hip_average_min_deg"],
                                knee_standing_min=bc["knee_average_min_deg"])
            if qa["quality"] != "PASS": continue
            _, pose2d = zips[obj.origin].read_2d(obj.seq, 1)
            start, end = ref.frame_range
            raw2d = to_common_skeleton(pose2d[start:end + 1])
            evidence = heel_motion_evidence(raw2d, no_max=hs["no_evidence_max"], yes_min=hs["positive_evidence_min"])
            # Same three-dimensional heel-only vector definition as the preceding diagnostic.
            from ai_trainer.offline_pose_benchmark import geometry_vector
            geometry, _ = geometry_vector(ref.coords, raw2d)
            phase = extract_phase_features(ref.coords)
            samples.append({"sample_id": f"{label}_{obj.origin}_{obj.seq.actor}_{obj.seq.level}_{obj.seq.rep}",
                            "truth": label, "actor": obj.seq.actor, "features": extract_all_features(ref.coords),
                            "bounds": segment_phases(phase).as_dict(), "evidence": evidence["verdict"],
                            "heel_score": evidence["value"], "heel_vector": geometry[12:]})
    for archive in zips.values(): archive.close()

    base_rows, _ = pairwise_dtw(samples, cfg, weights)
    direct = heel_direct_predictions(samples)
    expanded, policy_metrics = evaluate_rows(base_rows, direct)
    raw_candidates = []
    for i, row in enumerate(base_rows):
        if row["raw_prediction"] != HEEL: continue
        item = {**row, "heel_direct_prediction": direct[i],
                "heel_only_normal_distance": samples[i]["heel_only_normal_distance"],
                "heel_only_heel_distance": samples[i]["heel_only_heel_distance"],
                "heel_only_margin": samples[i]["heel_only_margin"]}
        for policy in POLICIES:
            item[policy + "_result"] = apply_policy(policy, row["raw_prediction"], row["heel_evidence"], direct[i])[0]
        raw_candidates.append(item)

    evidence_rows = evidence_table([{"truth": r["ground_truth"], "evidence": r["heel_evidence"],
                                     "raw_prediction": r["raw_prediction"]} for r in base_rows])
    empirical = empirical_evidence([{"truth": r["ground_truth"], "evidence": r["heel_evidence"]} for r in base_rows])
    heel_dist = [{"sample_id": s["sample_id"], "ground_truth": s["truth"],
                  "normal_distance": s["heel_only_normal_distance"], "heel_distance": s["heel_only_heel_distance"],
                  "relative_margin": s["heel_only_margin"], "prediction": direct[i]}
                 for i, s in enumerate(samples)]

    heel_candidate_margins = sorted(abs(r["relative_margin"]) for r in raw_candidates)
    weak_cut = float(np.quantile(heel_candidate_margins, .33)) if heel_candidate_margins else 0.0
    strong_cut = float(np.quantile(heel_candidate_margins, .67)) if heel_candidate_margins else 0.0
    matrix_counter = Counter()
    for row in base_rows:
        band = preference_band(row["relative_margin"], weak_cut, strong_cut)
        matrix_counter[(band, row["heel_evidence"], row["ground_truth"])] += 1
    joint_matrix = [{"dtw_preference": b, "heel_evidence": e,
                     "n": matrix_counter[(b,e,NORMAL)] + matrix_counter[(b,e,HEEL)],
                     "true_normal": matrix_counter[(b,e,NORMAL)], "true_heel": matrix_counter[(b,e,HEEL)]}
                    for b in ("Strong Heel","Weak Heel","Tie/Ambiguous","Normal preference")
                    for e in ("NO","AMBIGUOUS","YES")]

    # Current production code is Policy B: NO is rejected, AMBIGUOUS/YES are retained.
    tradeoff = [{k: row[k] for k in ("policy", "normal_to_heel_fp", "heel_to_unknown", "normal_to_unknown",
                                     "true_heel_preserved", "heel_precision", "heel_recall", "unknown_rate")}
                for row in policy_metrics]
    margin_rows = [{"sample_id": r["sample_id"], "ground_truth": r["ground_truth"],
                    "heel_evidence": r["heel_evidence"], "heel_score": r["heel_score"],
                    "normal_distance": r["normal_distance"], "heel_distance": r["heel_distance"],
                    "relative_margin": r["relative_margin"],
                    "preference_band": preference_band(r["relative_margin"], weak_cut, strong_cut)} for r in raw_candidates]

    latest_rebuild = max((p for p in (ROOT / "output/diagnostics/quality_gated_reference_rebuild").iterdir()
                          if (p / "virtual_reference").exists()), key=lambda p: p.name)
    clean_bench = OfflinePoseBenchmark(ROOT, db_dir=latest_rebuild / "virtual_reference")
    clean = [s for s in clean_bench.samples if s["label"] in (NORMAL, HEEL)]
    secondary = {"n": len(clean), "note": "Secondary clean operational context; primary policy metrics use 69 PASS sources."}

    after = {str(p): file_state(p) for p in db_paths}
    summary = {
        "evaluation_name": "Quality-PASS Source Offline Shadow Evaluation",
        "dataset": dict(Counter(s["truth"] for s in samples)), "evaluated_n": len(samples),
        "thresholds_unchanged": hs, "current_production_guard_equivalent": "B_NO_REJECT",
        "policy_metrics": policy_metrics, "empirical_evidence": empirical,
        "raw_heel_candidates_n": len(raw_candidates), "preference_band_quantiles_descriptive_only": {"weak_cut_q33": weak_cut, "strong_cut_q67": strong_cut},
        "strong_heel_and_no_samples": [r["sample_id"] for r in margin_rows if r["preference_band"] == "Strong Heel" and r["heel_evidence"] == "NO"],
        "secondary_clean_operational": secondary, "reference_db_before": before, "reference_db_after": after,
        "reference_db_unchanged": before == after, "production_runtime_cost": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_csv(out / "policy_metrics.csv", policy_metrics)
    write_csv(out / "heel_tradeoff.csv", tradeoff)
    write_csv(out / "raw_heel_candidates.csv", raw_candidates)
    write_csv(out / "evidence_confusion.csv", evidence_rows)
    write_csv(out / "evidence_empirical_proportions.csv", empirical)
    write_csv(out / "heel_only_distance_analysis.csv", heel_dist)
    write_csv(out / "dtw_margin_vs_evidence.csv", margin_rows)
    write_csv(out / "policy_comparison.csv", expanded)
    write_csv(out / "joint_dtw_evidence_matrix.csv", joint_matrix)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = ["# Heel Policy Shadow Evaluation", "", "Quality-PASS Source Offline Shadow Evaluation.", "",
              "Production classification changed: NO", "Production heel guard changed: NO",
              "Production DTW changed: NO", "Production thresholds changed: NO",
              "Reference DB changed: NO", "UI changed: NO", "Production runtime cost: 0", "",
              "## Policy metrics", ""]
    for row in policy_metrics:
        report.append(f"- {row['policy']}: accuracy={row['accuracy']:.4f}, heel precision={row['heel_precision']:.4f}, heel recall={row['heel_recall']:.4f}, normal→heel={row['normal_to_heel_fp']}, unknown={row['unknown_rate']:.4f}")
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
