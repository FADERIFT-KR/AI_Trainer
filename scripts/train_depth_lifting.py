#!/usr/bin/env python3
"""Train and select the 27-frame depth-sensitive VideoPose3D variant."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from ai_trainer.squat.depth_lifting import VideoPose3DDepthNet, depth_aware_lifting_loss  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output" / "depth_lifting_dataset"
OUT_DIR = ROOT / "output" / "depth_lifting_v2"
EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3
SEED = 20260919
TARGET_UNIT_MM = 1000.0


def _load(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    with np.load(DATA_DIR / f"{split}.npz") as data:
        # Regress in metres rather than raw millimetres.  This does not alter
        # geometry; it puts coordinate and bone losses in a numerically stable
        # range for AdamW.
        return torch.from_numpy(data["x_norm"]).float(), torch.from_numpy(data["y_norm"] / TARGET_UNIT_MM).float()


def _metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    delta = (pred - target) * TARGET_UNIT_MM
    axis_mae = delta.abs().mean(dim=(0, 1))
    return {
        "mpjpe_mm": float(torch.linalg.vector_norm(delta, dim=-1).mean()),
        "x_mae_mm": float(axis_mae[0]),
        "y_mae_mm": float(axis_mae[1]),
        "z_mae_mm": float(axis_mae[2]),
    }


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, y_train = _load("train")
    x_val, y_val = _load("val")
    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=512, shuffle=False)
    model = VideoPose3DDepthNet(n_joints=18, channels=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_z = float("inf")
    started = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            prediction = model(xb.to(device))
            loss, _ = depth_aware_lifting_loss(prediction, yb.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        predictions: list[torch.Tensor] = []
        targets: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                predictions.append(model(xb.to(device)).cpu())
                targets.append(yb)
        metrics = _metrics(torch.cat(predictions), torch.cat(targets))
        metrics["epoch"] = epoch
        history.append(metrics)
        print(f"epoch {epoch:02d}/{EPOCHS}: MPJPE={metrics['mpjpe_mm']:.2f} mm, z-MAE={metrics['z_mae_mm']:.2f} mm")
        if metrics["z_mae_mm"] < best_z:
            best_z = metrics["z_mae_mm"]
            torch.save(model.state_dict(), OUT_DIR / "model_best_z.pt")
        scheduler.step()
    report = {
        "architecture": "VideoPose3D-style residual dilated TCN (filter widths 3,3,3; 27-frame symmetric context)",
        "temporal_window": 27,
        "n_params": sum(p.numel() for p in model.parameters()),
        "loss": "metre-space x/y MSE + 2*z MSE + 0.25*bone-length MSE",
        "checkpoint_output_unit": "metres (multiply predictions by 1000 for mm)",
        "selection_metric": "validation z-MAE (mm)",
        "actor_split_preserved": True,
        "seed": SEED,
        "epochs": EPOCHS,
        "elapsed_sec": time.time() - started,
        "history": history,
        "best": min(history, key=lambda item: float(item["z_mae_mm"])),
        "deployment_note": "This checkpoint is validated on AI Hub 2D/triangulated-3D only. Do not replace MediaPipe world landmarks for live judgement until real-camera 3D validation is available.",
    }
    (OUT_DIR / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {OUT_DIR / 'model_best_z.pt'}")


if __name__ == "__main__":
    main()
