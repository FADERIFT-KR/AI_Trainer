#!/usr/bin/env python3
"""Train the camera-ray T=9 lifting ablation on actor-disjoint AIHub data."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from ai_trainer.depth_lifting import depth_aware_lifting_loss  # noqa: E402
from ai_trainer.ray_lifting import RAY_FEATURE_TYPE, ProjectiveCamera, RayTemporalLiftingNet  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output" / "ray_lifting_dataset"
OUT_DIR = ROOT / "output" / "ray_lifting_camera1"
TARGET_UNIT_MM = 1000.0
SEED = 20260920


def _device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load(split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with np.load(DATA_DIR / f"{split}.npz") as data:
        # Train in metres for numerically stable coordinate and bone losses.
        return (
            torch.from_numpy(data["x_ray"]).float(),
            torch.from_numpy(data["y_camera"] / TARGET_UNIT_MM).float(),
            torch.from_numpy(data["y_world"] / TARGET_UNIT_MM).float(),
        )


def _metrics(
    prediction_camera: torch.Tensor,
    target_camera: torch.Tensor,
    target_world: torch.Tensor,
    rotation_world_to_camera: torch.Tensor,
) -> dict[str, float]:
    delta_camera_mm = (prediction_camera - target_camera) * TARGET_UNIT_MM
    prediction_world = prediction_camera @ rotation_world_to_camera
    delta_world_mm = (prediction_world - target_world) * TARGET_UNIT_MM
    camera_axis_mae = delta_camera_mm.abs().mean(dim=(0, 1))
    world_axis_mae = delta_world_mm.abs().mean(dim=(0, 1))
    return {
        "mpjpe_mm": float(torch.linalg.vector_norm(delta_world_mm, dim=-1).mean()),
        "world_x_mae_mm": float(world_axis_mae[0]),
        "world_y_mae_mm": float(world_axis_mae[1]),
        "world_z_mae_mm": float(world_axis_mae[2]),
        "camera_x_mae_mm": float(camera_axis_mae[0]),
        "camera_y_mae_mm": float(camera_axis_mae[1]),
        "camera_depth_z_mae_mm": float(camera_axis_mae[2]),
    }


def _evaluate(
    model: RayTemporalLiftingNet,
    loader: DataLoader,
    device: torch.device,
    rotation: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    predictions: list[torch.Tensor] = []
    camera_targets: list[torch.Tensor] = []
    world_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for rays, camera_target, world_target in loader:
            predictions.append(model(rays.to(device)).cpu())
            camera_targets.append(camera_target)
            world_targets.append(world_target)
    return _metrics(
        torch.cat(predictions), torch.cat(camera_targets), torch.cat(world_targets), rotation
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.lr <= 0.0:
        raise SystemExit("epochs, batch-size, and lr must be positive")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = _device(args.device)
    train_rays, train_camera, train_world = _load("train")
    val_rays, val_camera, val_world = _load("val")
    if train_rays.shape[1:] != (9, 18, 3):
        raise RuntimeError(f"unexpected ray feature shape: {tuple(train_rays.shape)}")
    camera = ProjectiveCamera.load(DATA_DIR / "camera1_projective.json")
    rotation = torch.from_numpy(camera.rotation_world_to_camera.astype(np.float32))
    train_loader = DataLoader(
        TensorDataset(train_rays, train_camera, train_world),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_rays, val_camera, val_world), batch_size=512, shuffle=False
    )
    model = RayTemporalLiftingNet(n_joints=18, hidden=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_depth = float("inf")
    history: list[dict[str, float | int]] = []
    started = time.time()

    print(f"device={device}; train={tuple(train_rays.shape)}; val={tuple(val_rays.shape)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for rays, target, _world in train_loader:
            prediction = model(rays.to(device))
            loss, _components = depth_aware_lifting_loss(prediction, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        metrics = _evaluate(model, val_loader, device, rotation)
        metrics["epoch"] = epoch
        history.append(metrics)
        print(
            f"epoch {epoch:02d}/{args.epochs}: MPJPE={metrics['mpjpe_mm']:.2f} mm, "
            f"world-z={metrics['world_z_mae_mm']:.2f} mm, "
            f"camera-depth-z={metrics['camera_depth_z_mae_mm']:.2f} mm"
        )
        if metrics["camera_depth_z_mae_mm"] < best_depth:
            best_depth = metrics["camera_depth_z_mae_mm"]
            torch.save(model.state_dict(), OUT_DIR / "model_best_camera_depth.pt")
        scheduler.step()

    best = min(history, key=lambda item: float(item["camera_depth_z_mae_mm"]))
    report = {
        "feature_type": RAY_FEATURE_TYPE,
        "architecture": "TemporalLiftingNet, T=9, ray_x/ray_y/ray_z per joint",
        "input": "unit camera rays from non-mirrored undistorted pixels",
        "target": "Hip-relative camera-coordinate 3D, metres during training",
        "checkpoint_output_unit": "metres (multiply by 1000 for mm)",
        "selection_metric": "held-out actor camera-depth z-MAE (mm)",
        "actor_split_preserved": True,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_sec": time.time() - started,
        "best": best,
        "history": history,
        "deployment_note": (
            "Offline AIHub ray ablation only. The live pipeline remains on MediaPipe world landmarks "
            "until paired, real-webcam 3D validation demonstrates improvement."
        ),
    }
    (OUT_DIR / "train_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "feature_type": RAY_FEATURE_TYPE,
                "temporal_window": 9,
                "joint_count": 18,
                "input_dimensions": 3,
                "input_pixel_convention": "undistorted, physical/non-mirrored camera pixels",
                "target_coordinate_system": "Hip-relative camera coordinates",
                "unit": "metres",
                "camera_projection": str((DATA_DIR / "camera1_projective.json").relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved: {OUT_DIR / 'model_best_camera_depth.pt'}")


if __name__ == "__main__":
    main()
