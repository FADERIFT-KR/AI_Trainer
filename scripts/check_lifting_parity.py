"""Compare current live normalization with stored offline lifting input."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.lifting_model import TemporalLiftingNet
from ai_trainer.lifting_parity import (
    finalize_single_3d,
    live_preprocess_2d,
    mean_leg_angles,
    pelvis_height,
)

def main() -> None:
    dataset = np.load(ROOT / "output" / "lifting_dataset" / "val.npz")
    raw = dataset["x_raw"][0]
    offline = dataset["x_norm"][0].astype(np.float32)
    live, live_scale = live_preprocess_2d(raw)

    hip = COMMON_JOINT_NAMES.index("Hip")
    neck = COMMON_JOINT_NAMES.index("Neck")
    offline_torso = np.linalg.norm(offline[:, neck] - offline[:, hip], axis=-1)
    raw_torso = np.linalg.norm(raw[:, neck] - raw[:, hip], axis=-1)
    offline_scale = float(np.median(raw_torso / np.maximum(offline_torso, 1e-8)))

    device = torch.device("cpu")
    model = TemporalLiftingNet(n_joints=18, hidden=128)
    checkpoint = ROOT / "output" / "lifting_baseline" / "model_best.pt"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    with torch.no_grad():
        live_3d = model(torch.from_numpy(live[None]).to(device))[0].cpu().numpy()
        offline_3d = model(torch.from_numpy(offline[None]).to(device))[0].cpu().numpy()
        repeat_3d = model(torch.from_numpy(offline[None]).to(device))[0].cpu().numpy()

    live_final = finalize_single_3d(live_3d)
    offline_final = finalize_single_3d(offline_3d)
    live_hip, live_knee = mean_leg_angles(live_final)
    offline_hip, offline_knee = mean_leg_angles(offline_final)

    input_diff = np.abs(live - offline)
    output_diff = np.abs(live_3d - offline_3d)
    repeat_diff = np.abs(repeat_3d - offline_3d)
    result = {
        "sample_index": 0,
        "center_frame": int(dataset["center_frame"][0]),
        "joint_order": COMMON_JOINT_NAMES,
        "input_shape_live": list(live.shape),
        "input_shape_offline": list(offline.shape),
        "input_dtype_live": str(live.dtype),
        "input_dtype_offline": str(offline.dtype),
        "live_scale_first_8": live_scale,
        "offline_scale_full_clip": offline_scale,
        "scale_difference": live_scale - offline_scale,
        "root_max_abs_live": float(np.abs(live[:, hip]).max()),
        "root_max_abs_offline": float(np.abs(offline[:, hip]).max()),
        "input_range_live": [float(live.min()), float(live.max())],
        "input_range_offline": [float(offline.min()), float(offline.max())],
        "input_max_abs_diff": float(input_diff.max()),
        "input_mean_abs_diff": float(input_diff.mean()),
        "model_output_max_abs_diff": float(output_diff.max()),
        "model_output_mean_abs_diff": float(output_diff.mean()),
        "same_tensor_repeat_max_abs_diff": float(repeat_diff.max()),
        "hip_angle_live": live_hip,
        "hip_angle_offline": offline_hip,
        "hip_angle_abs_diff": abs(live_hip - offline_hip),
        "knee_angle_live": live_knee,
        "knee_angle_offline": offline_knee,
        "knee_angle_abs_diff": abs(live_knee - offline_knee),
        "pelvis_live": pelvis_height(live_final),
        "pelvis_offline": pelvis_height(offline_final),
        "pelvis_abs_diff": abs(pelvis_height(live_final) - pelvis_height(offline_final)),
        "parity_pass": bool(np.allclose(live, offline, atol=1e-6, rtol=1e-6)),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
