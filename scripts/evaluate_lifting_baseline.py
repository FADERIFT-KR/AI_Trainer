#!/usr/bin/env python3
"""승인된 baseline checkpoint(model_best.pt)를 평가하고 재현 가능성을 위한
전체 메타데이터를 기록한다. 모델을 다시 학습하지 않는다 (튜닝 금지 지침 준수).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES  # noqa: E402
from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output" / "lifting_dataset"
OUT_DIR = ROOT / "output" / "lifting_baseline"


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = TemporalLiftingNet(n_joints=18, hidden=128)
    model.load_state_dict(torch.load(OUT_DIR / "model_best.pt", map_location=device))
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    d = np.load(DATA_DIR / "val.npz")
    x_val = torch.from_numpy(d["x_norm"]).float().to(device)
    y_val = torch.from_numpy(d["y_norm"]).float().to(device)

    with torch.no_grad():
        preds = []
        for i in range(0, x_val.shape[0], 1024):
            preds.append(model(x_val[i : i + 1024]))
        pred = torch.cat(preds, dim=0)

    per_joint_err = torch.norm(pred - y_val, dim=-1).mean(dim=0).cpu().numpy()  # (18,)
    overall_mpjpe = float(per_joint_err.mean())

    train_report = json.loads((OUT_DIR / "train_report.json").read_text(encoding="utf-8"))

    metadata = {
        "checkpoint": str(OUT_DIR / "model_best.pt"),
        "model": {
            "architecture": "TemporalLiftingNet (dilated Conv1d x3 [dilation=1,2,1] + FC head)",
            "n_params": n_params,
            "temporal_window_T": 9,
            "hidden_dim": 128,
        },
        "data": {
            "n_train": train_report["n_train"],
            "n_val": train_report["n_val"],
            "actor_split_file": "configs/actor_split.json",
            "camera": 1,
            "frame_stride": 2,
        },
        "normalization": {
            "input_2d": "per-frame Hip root-center + per-sequence median(Neck-Hip 2D length) scale",
            "target_3d": "per-frame Hip-centered translation only (raw mm scale, no scale/orientation norm)",
        },
        "training": {
            "epochs": train_report["epochs"],
            "batch_size": train_report["batch_size"],
            "lr": train_report["lr"],
            "optimizer": "Adam",
            "loss": "MSE",
            "random_seed": "미고정 (이번 run은 명시적 seed 설정 없이 실행됨 — 향후 재학습 시 seed 고정 예정, 재현성 caveat로 기록)",
            "elapsed_sec": train_report["elapsed_sec"],
        },
        "evaluation": {
            "protocol": "MPJPE, Human3.6M Protocol-1과 동일하게 root(Hip)-align만 적용, mm 단위",
            "trivial_zero_baseline_val_mpjpe_mm": train_report["trivial_zero_val_mpjpe_mm"],
            "best_val_mpjpe_mm": train_report["best_val_mpjpe_mm"],
            "val_mpjpe_recomputed_mm": overall_mpjpe,
            "per_joint_mpjpe_mm": {
                name: float(err) for name, err in zip(COMMON_JOINT_NAMES, per_joint_err)
            },
        },
        "note": "추가 튜닝 금지 지침에 따라 이 checkpoint를 baseline으로 확정하고 보존한다. "
        "T=27/81 확장, weight decay, LR scheduler, dropout, depth/width 조정은 "
        "end-to-end 파이프라인 완성 후 별도 ablation 단계에서 검토한다.",
    }

    out_path = OUT_DIR / "baseline_metadata.json"
    out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"val MPJPE (재확인): {overall_mpjpe:.2f} mm")
    print("\nper-joint MPJPE (mm):")
    for name, err in sorted(zip(COMMON_JOINT_NAMES, per_joint_err), key=lambda kv: -kv[1]):
        print(f"  {name:12s} {err:6.2f}")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
