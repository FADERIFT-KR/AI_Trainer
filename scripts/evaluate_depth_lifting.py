#!/usr/bin/env python3
"""Report per-axis validation error for the old and long-context lifting models."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.squat.depth_lifting import VideoPose3DDepthNet  # noqa: E402
from ai_trainer.squat.lifting_model import TemporalLiftingNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def _metrics(model: torch.nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            prediction = model(torch.from_numpy(x[start : start + 512]).float())
            predictions.append(prediction)
    delta = torch.cat(predictions) - torch.from_numpy(y).float()
    mae = delta.abs().mean(dim=(0, 1))
    return {
        "mpjpe_mm": float(torch.linalg.vector_norm(delta, dim=-1).mean()),
        "x_mae_mm": float(mae[0]),
        "y_mae_mm": float(mae[1]),
        "z_mae_mm": float(mae[2]),
    }


def main() -> None:
    with np.load(ROOT / "output" / "lifting_dataset" / "val.npz") as old_data:
        old_x, old_y = old_data["x_norm"], old_data["y_norm"]
    old = TemporalLiftingNet(n_joints=18, hidden=128)
    old.load_state_dict(torch.load(ROOT / "output" / "lifting_baseline" / "model_best.pt", map_location="cpu", weights_only=True))
    with np.load(ROOT / "output" / "depth_lifting_dataset" / "val.npz") as long_data:
        long_x, long_y = long_data["x_norm"], long_data["y_norm"]
    long = VideoPose3DDepthNet(n_joints=18, channels=128)
    long.load_state_dict(torch.load(ROOT / "output" / "depth_lifting_v2" / "model_best_z.pt", map_location="cpu", weights_only=True))
    # V2 is trained in metres for numerical stability; all reports remain mm.
    def _v2_metrics() -> dict[str, float]:
        long.eval()
        predictions = []
        with torch.no_grad():
            for start in range(0, len(long_x), 512):
                predictions.append(long(torch.from_numpy(long_x[start:start + 512]).float()) * 1000.0)
        delta = torch.cat(predictions) - torch.from_numpy(long_y).float()
        mae = delta.abs().mean(dim=(0, 1))
        return {"mpjpe_mm": float(torch.linalg.vector_norm(delta, dim=-1).mean()), "x_mae_mm": float(mae[0]), "y_mae_mm": float(mae[1]), "z_mae_mm": float(mae[2])}

    result = {
        "baseline_9_frame": _metrics(old, old_x, old_y),
        "v2_27_frame": _v2_metrics(),
        "caveat": "Both evaluations use the same held-out actors but not identical temporal windows. Compare z-MAE as a deployment-selection signal, not as a claim of metric real-webcam depth accuracy.",
    }
    result["z_mae_change_mm"] = result["v2_27_frame"]["z_mae_mm"] - result["baseline_9_frame"]["z_mae_mm"]
    path = ROOT / "output" / "depth_lifting_v2" / "validation_comparison.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # A longer temporal model must demonstrably improve depth on unseen actors
    # before it can be considered.  Keep this decision machine-readable so a
    # later retraining cannot accidentally replace the safer baseline.
    selection = {
        "selected_model": "v2_27_frame" if result["z_mae_change_mm"] < 0.0 else "baseline_9_frame",
        "reason": "held-out z-MAE improved" if result["z_mae_change_mm"] < 0.0 else "V2 did not improve held-out z-MAE",
        "baseline_z_mae_mm": result["baseline_9_frame"]["z_mae_mm"],
        "candidate_z_mae_mm": result["v2_27_frame"]["z_mae_mm"],
        "live_source": "MediaPipe world landmarks (unchanged; neither AI Hub-only lifting model is auto-promoted to live webcam input)",
    }
    (ROOT / "output" / "depth_lifting_v2" / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
