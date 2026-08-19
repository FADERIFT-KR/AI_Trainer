#!/usr/bin/env python3
"""2D->3D Temporal Lifting baseline 학습 + MPJPE 평가.

입력 : x_norm (B,T=9,J=18,2)  -- camera1 정규화된 2D
타깃 : y_norm (B,J=18,3)       -- Hip-centered 3D GT (raw mm 스케일)
평가 : MPJPE(mm) = mean(||pred - gt||_2), Human3.6M Protocol-1과 동일하게 root-align만 적용.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from ai_trainer.lifting_model import TemporalLiftingNet  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "output" / "lifting_dataset"
OUT_DIR = ROOT / "output" / "lifting_baseline"

EPOCHS = 30
BATCH_SIZE = 256
LR = 1e-3


def load_tensors(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    d = np.load(DATA_DIR / f"{split}.npz")
    x = torch.from_numpy(d["x_norm"]).float()  # (N,T,18,2)
    y = torch.from_numpy(d["y_norm"]).float()  # (N,18,3)
    return x, y


def mpjpe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    return torch.norm(pred - gt, dim=-1).mean().item()


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device = {device}")

    x_train, y_train = load_tensors("train")
    x_val, y_val = load_tensors("val")
    print(f"train: {tuple(x_train.shape)} -> {tuple(y_train.shape)}")
    print(f"val  : {tuple(x_val.shape)} -> {tuple(y_val.shape)}")

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=512, shuffle=False)

    model = TemporalLiftingNet(n_joints=18, hidden=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"모델 파라미터 수: {n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # baseline 비교용: "예측 = 항상 0(=Hip 위치)"일 때의 MPJPE (아무것도 안 배운 모델 대비 개선폭 확인용)
    zero_mpjpe_val = torch.norm(y_val, dim=-1).mean().item()
    print(f"[참고] 항상 0(Hip)을 예측하는 trivial baseline의 val MPJPE = {zero_mpjpe_val:.2f} mm")

    history = []
    best_val_mpjpe = float("inf")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_sum, train_mpjpe_sum, n_train = 0.0, 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            bs = xb.shape[0]
            train_loss_sum += loss.item() * bs
            train_mpjpe_sum += mpjpe(pred, yb) * bs
            n_train += bs

        model.eval()
        val_mpjpe_sum, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                bs = xb.shape[0]
                val_mpjpe_sum += mpjpe(pred, yb) * bs
                n_val += bs

        train_loss = train_loss_sum / n_train
        train_mpjpe = train_mpjpe_sum / n_train
        val_mpjpe = val_mpjpe_sum / n_val
        history.append({"epoch": epoch, "train_loss": train_loss, "train_mpjpe": train_mpjpe, "val_mpjpe": val_mpjpe})
        print(f"epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:9.2f}  train_MPJPE={train_mpjpe:6.2f}mm  val_MPJPE={val_mpjpe:6.2f}mm")

        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            torch.save(model.state_dict(), OUT_DIR / "model_best.pt")

    elapsed = time.time() - t0
    print(f"\n학습 완료 ({elapsed:.1f}초). best val MPJPE = {best_val_mpjpe:.2f} mm  (trivial baseline {zero_mpjpe_val:.2f} mm)")

    report = {
        "n_params": n_params,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "trivial_zero_val_mpjpe_mm": zero_mpjpe_val,
        "best_val_mpjpe_mm": best_val_mpjpe,
        "elapsed_sec": elapsed,
        "history": history,
    }
    (OUT_DIR / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT_DIR / 'model_best.pt'}, {OUT_DIR / 'train_report.json'}")


if __name__ == "__main__":
    main()
