#!/usr/bin/env python3
"""Build a camera-ray lifting dataset without leaking validation 3D labels.

The AIHub release used here does not include camera1 K/R/t metadata.  We first
estimate an *audit-only* pinhole camera from training actors' paired 2D/3D
labels, then convert the original, non-mirrored camera1 pixels to unit rays.
The estimate is saved with its train/validation reprojection statistics.

This makes a reproducible Ray3D-style ablation possible, but it is not a
substitute for ChArUco calibration of a user's webcam.  The resulting model is
therefore not automatically connected to the live judgement path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ai_trainer.ray_lifting import (  # noqa: E402
    RAY_FEATURE_TYPE,
    ProjectiveCamera,
    estimate_projective_camera,
)


ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "output" / "lifting_dataset"
OUT_DIR = ROOT / "output" / "ray_lifting_dataset"
CAMERA_PATH = OUT_DIR / "camera1_projective.json"
METADATA_PATH = OUT_DIR / "metadata.json"


def _error_summary(errors: np.ndarray) -> dict[str, float]:
    return {
        "mean_px": float(errors.mean()),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.quantile(errors, 0.95)),
    }


def _arrays_for_split(source: np.lib.npyio.NpzFile, camera: ProjectiveCamera) -> dict[str, np.ndarray]:
    raw_pixels = source["x_raw"].astype(np.float64)
    world_relative = source["y_norm"].astype(np.float64)
    rays = camera.pixels_to_rays(raw_pixels).astype(np.float32)
    # y_norm is Hip-relative world-3D.  Rotate it into the camera axes so the
    # model's z error means optical-depth error rather than an arbitrary dataset
    # global-axis error.  Translation is intentionally omitted.
    camera_relative = camera.world_relative_to_camera(world_relative).astype(np.float32)
    result = {
        "x_ray": rays,
        "y_camera": camera_relative,
        "y_world": world_relative.astype(np.float32),
    }
    for name in ("origin", "actor", "level", "error_type", "rep", "center_frame"):
        result[name] = source[name]
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE_DIR / "train.npz") as train_source:
        train_world = train_source["y_raw"].reshape(-1, 3)
        train_pixels = train_source["x_raw"][:, train_source["x_raw"].shape[1] // 2].reshape(-1, 2)
        camera = estimate_projective_camera(train_world, train_pixels)
        train_actor_set = set(train_source["actor"].astype(str))
        train_arrays = _arrays_for_split(train_source, camera)
        train_errors = camera.reprojection_errors(train_world, train_pixels)

    with np.load(SOURCE_DIR / "val.npz") as val_source:
        val_actor_set = set(val_source["actor"].astype(str))
        overlap = train_actor_set & val_actor_set
        if overlap:
            raise RuntimeError(f"actor split leakage: {sorted(overlap)}")
        val_arrays = _arrays_for_split(val_source, camera)
        val_world = val_source["y_raw"].reshape(-1, 3)
        val_pixels = val_source["x_raw"][:, val_source["x_raw"].shape[1] // 2].reshape(-1, 2)
        val_errors = camera.reprojection_errors(val_world, val_pixels)

    np.savez_compressed(OUT_DIR / "train.npz", **train_arrays)
    np.savez_compressed(OUT_DIR / "val.npz", **val_arrays)
    camera.save(CAMERA_PATH)
    metadata = {
        "feature_type": RAY_FEATURE_TYPE,
        "input": "camera1 raw, undistorted/non-mirrored pixel -> unit ray K^-1[u,v,1]",
        "target": "Hip-relative 3D, rotated from dataset world axes into estimated camera axes, mm",
        "temporal_window": int(train_arrays["x_ray"].shape[1]),
        "joint_count": int(train_arrays["x_ray"].shape[2]),
        "n_train": int(len(train_arrays["x_ray"])),
        "n_val": int(len(val_arrays["x_ray"])),
        "actor_split_preserved": True,
        "train_actor_count": len(train_actor_set),
        "val_actor_count": len(val_actor_set),
        "camera_estimation": {
            "fit_from": "training actors only; centre-frame camera1 2D/3D correspondences",
            "camera_file": CAMERA_PATH.name,
            "train_reprojection_error_px": _error_summary(train_errors),
            "held_out_actor_reprojection_error_px": _error_summary(val_errors),
            "limitation": (
                "AIHub did not provide physical camera metadata. This dataset-derived projection "
                "is for an offline ray-feature ablation and must not be treated as a user-webcam calibration."
            ),
        },
        "live_deployment": "disabled pending paired real-webcam 2D/3D validation",
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"train {train_arrays['x_ray'].shape} -> {train_arrays['y_camera'].shape}")
    print(f"val   {val_arrays['x_ray'].shape} -> {val_arrays['y_camera'].shape}")
    print("reprojection px:", json.dumps(metadata["camera_estimation"], ensure_ascii=False))
    print(f"saved: {OUT_DIR}")


if __name__ == "__main__":
    main()
