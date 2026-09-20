#!/usr/bin/env python3
"""Measure AI Hub 3-D skeleton jitter/spike refinement on every sequence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ai_trainer.actor_split import load_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip, JOINT_NAMES  # noqa: E402
from ai_trainer.common_skeleton import to_common_skeleton  # noqa: E402
from ai_trainer.dataset_config import DATASET_PATH  # noqa: E402
from ai_trainer.normalization import hip_center_3d, leg_length_scale, orientation_align_3d  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.phase_segmentation import segment_phases  # noqa: E402
from ai_trainer.skeleton_filter import SkeletonFilterConfig, refine_skeleton_sequence  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "configs" / "skeleton_refinement_report.json"


def _motion_features(coords_26: np.ndarray):
    coords = to_common_skeleton(coords_26)
    centered, _ = hip_center_3d(coords)
    scale = max(float(np.median(leg_length_scale(centered))), 1e-8)
    aligned = orientation_align_3d(centered / scale, reference_frames=5)
    features = extract_phase_features(aligned)
    phases = segment_phases(features)
    boundaries = np.asarray(
        [phases.prep_end, phases.bottom_start, phases.bottom_end, phases.rise_end],
        dtype=np.float64,
    )
    return features, boundaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    sequences = load_air_squat_sequences(args.dataset)
    if args.limit is not None:
        sequences = sequences[: max(0, args.limit)]
    cfg = SkeletonFilterConfig()
    reports = []
    with AiHubZip(args.dataset) as dataset:
        for index, item in enumerate(sequences, start=1):
            _, raw = dataset.read_3d(item.seq, refine=False)
            refined, report = refine_skeleton_sequence(raw, JOINT_NAMES, cfg)
            raw_features, raw_bounds = _motion_features(raw)
            refined_features, refined_bounds = _motion_features(refined)
            report_dict = report.to_dict()
            report_dict["phase_boundary_mae_frames"] = float(np.mean(np.abs(raw_bounds - refined_bounds)))
            report_dict["bottom_center_shift_frames"] = float(
                abs(np.mean(raw_bounds[1:3]) - np.mean(refined_bounds[1:3]))
            )
            report_dict["knee_angle_rms_change_deg"] = float(
                np.sqrt(np.mean((raw_features.knee_flexion_deg - refined_features.knee_flexion_deg) ** 2))
            )
            reports.append(report_dict)
            if index % 50 == 0:
                print(f"  ...{index}/{len(sequences)}")
    if not reports:
        raise RuntimeError("평가할 3D 시퀀스가 없습니다.")

    aggregate = {}
    for key in reports[0]:
        values = np.asarray([report[key] for report in reports], dtype=np.float64)
        aggregate[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
        }
    raw_jerk = aggregate["raw_jerk_rms_normalized"]["mean"]
    refined_jerk = aggregate["refined_jerk_rms_normalized"]["mean"]
    raw_bone = aggregate["raw_bone_length_cv"]["mean"]
    refined_bone = aggregate["refined_bone_length_cv"]["mean"]
    payload = {
        "schema_version": 1,
        "dataset": str(args.dataset),
        "sequence_count": len(reports),
        "config": cfg.__dict__,
        "aggregate": aggregate,
        "improvement": {
            "jerk_rms_reduction": 1.0 - refined_jerk / max(raw_jerk, 1e-12),
            "bone_length_cv_reduction": 1.0 - refined_bone / max(raw_bone, 1e-12),
        },
        "method": [
            "rolling median/MAD spike detection and interpolation",
            "zero-phase 4th-order Butterworth low-pass filtering",
            "soft robust-median bone-length projection",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload["improvement"], ensure_ascii=False, indent=2))
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
