#!/usr/bin/env python3
"""Compare the ray-feature candidate with the existing T=9 2D baseline.

The comparison is actor-disjoint and uses identical centre frames.  It is an
AIHub offline ablation, not permission to replace the live MediaPipe 3D path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402
from ai_trainer.ray_lifting import ProjectiveCamera, RayTemporalLiftingNet  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "output" / "lifting_baseline"
SOURCE_DIR = ROOT / "output" / "lifting_dataset"
RAY_DATA_DIR = ROOT / "output" / "ray_lifting_dataset"
RAY_DIR = ROOT / "output" / "ray_lifting_camera1"
MM = 1000.0


def _axis_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    delta = prediction - target
    mae = delta.abs().mean(dim=(0, 1))
    return {
        "mpjpe_mm": float(torch.linalg.vector_norm(delta, dim=-1).mean()),
        "x_mae_mm": float(mae[0]),
        "y_mae_mm": float(mae[1]),
        "z_mae_mm": float(mae[2]),
    }


def _batched_predict(model: torch.nn.Module, data: np.ndarray) -> torch.Tensor:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(data), 512):
            outputs.append(model(torch.from_numpy(data[start : start + 512]).float()))
    return torch.cat(outputs)


def main() -> None:
    baseline = TemporalLiftingNet(n_joints=18, hidden=128).eval()
    baseline.load_state_dict(
        torch.load(BASELINE_DIR / "model_best.pt", map_location="cpu", weights_only=True)
    )
    with np.load(SOURCE_DIR / "val.npz") as source:
        baseline_prediction = _batched_predict(baseline, source["x_norm"])
        baseline_target = torch.from_numpy(source["y_norm"]).float()
    baseline_metrics = _axis_metrics(baseline_prediction, baseline_target)

    ray_model = RayTemporalLiftingNet(n_joints=18, hidden=128).eval()
    ray_model.load_state_dict(
        torch.load(RAY_DIR / "model_best_camera_depth.pt", map_location="cpu", weights_only=True)
    )
    camera = ProjectiveCamera.load(RAY_DATA_DIR / "camera1_projective.json")
    with np.load(RAY_DATA_DIR / "val.npz") as ray_data:
        ray_prediction_camera = _batched_predict(ray_model, ray_data["x_ray"])
        ray_target_camera = torch.from_numpy(ray_data["y_camera"]).float() / MM
        ray_target_world = torch.from_numpy(ray_data["y_world"]).float()
    # Candidate output is in metres and camera axes.  Map it back before
    # comparing to the legacy root-relative world-coordinate protocol.
    rotation = torch.from_numpy(camera.rotation_world_to_camera.astype(np.float32))
    ray_prediction_world_mm = ray_prediction_camera @ rotation * MM
    ray_metrics = _axis_metrics(ray_prediction_world_mm, ray_target_world)
    camera_delta_mm = (ray_prediction_camera - ray_target_camera) * MM
    camera_axis_mae = camera_delta_mm.abs().mean(dim=(0, 1))
    ray_metrics["camera_depth_z_mae_mm"] = float(camera_axis_mae[2])
    ray_metrics["camera_x_mae_mm"] = float(camera_axis_mae[0])
    ray_metrics["camera_y_mae_mm"] = float(camera_axis_mae[1])

    # Do not let a stale checkpoint and a later training report be compared as
    # though they were one experiment.  This caught an interrupted concurrent
    # training attempt during development and makes model selection auditable.
    train_report = json.loads((RAY_DIR / "train_report.json").read_text(encoding="utf-8"))
    recorded_depth = float(train_report["best"]["camera_depth_z_mae_mm"])
    if abs(ray_metrics["camera_depth_z_mae_mm"] - recorded_depth) > 1e-3:
        raise RuntimeError(
            "ray checkpoint and train_report disagree on validation camera-depth z-MAE; "
            "retrain or restore matching artifacts before evaluating"
        )

    better_global = (
        ray_metrics["mpjpe_mm"] < baseline_metrics["mpjpe_mm"]
        and ray_metrics["z_mae_mm"] < baseline_metrics["z_mae_mm"]
    )
    report = {
        "protocol": "AIHub camera1, actor-disjoint validation, identical T=9 centre frames",
        "baseline_2d_pixels": baseline_metrics,
        "candidate_camera_rays": ray_metrics,
        "checkpoint_report_consistent": True,
        "delta_candidate_minus_baseline_mm": {
            "mpjpe_mm": ray_metrics["mpjpe_mm"] - baseline_metrics["mpjpe_mm"],
            "global_z_mae_mm": ray_metrics["z_mae_mm"] - baseline_metrics["z_mae_mm"],
        },
        "offline_candidate_selected": better_global,
        "offline_selection_reason": (
            "candidate improves both world MPJPE and world z-MAE"
            if better_global
            else "candidate did not improve both world MPJPE and world z-MAE"
        ),
        "live_source": "MediaPipe world landmarks (unchanged)",
        "live_promotion": False,
        "live_promotion_reason": (
            "A dataset-derived camera projection is not a physical user-webcam calibration; "
            "paired real-webcam 2D/3D validation is still required."
        ),
    }
    RAY_DIR.mkdir(parents=True, exist_ok=True)
    (RAY_DIR / "validation_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (RAY_DIR / "selection.json").write_text(
        json.dumps(
            {
                "offline_selected_model": "ray_t9" if better_global else "baseline_2d_t9",
                "live_selected_source": "mediapipe_world_landmarks",
                "live_promotion": False,
                "reason": report["live_promotion_reason"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
