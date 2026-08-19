#!/usr/bin/env python3
"""2D->3D Temporal Lifting 학습 데이터셋을 생성해 npz로 저장한다.

actor_split.json 기준으로 시퀀스를 먼저 train/val로 나눈 뒤 윈도우 샘플을 만들며,
raw(정규화 전)와 norm(정규화 후) 좌표를 모두 저장해 감사(audit)할 수 있게 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from ai_trainer.lifting_dataset import build_split_datasets, samples_to_arrays  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Training/02.라벨링데이터/TL.zip"
)
VL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"
)
SPLIT_PATH = ROOT / "configs" / "actor_split.json"
OUT_DIR = ROOT / "output" / "lifting_dataset"


def main() -> None:
    print("데이터셋 생성 중 (camera1 2D + 3D GT, actor 단위 split)...")
    train_samples, val_samples = build_split_datasets(TL_ZIP, VL_ZIP, SPLIT_PATH)
    print(f"train 샘플: {len(train_samples)}개, val 샘플: {len(val_samples)}개")

    # 동일 actor가 양쪽에 없는지 최종 방어 검증
    train_actors = {s.actor for s in train_samples}
    val_actors = {s.actor for s in val_samples}
    overlap = train_actors & val_actors
    if overlap:
        raise RuntimeError(f"actor leakage 발견: {overlap}")
    print(f"actor 겹침 검사 통과 (train actor {len(train_actors)}명 / val actor {len(val_actors)}명)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, samples in [("train", train_samples), ("val", val_samples)]:
        arrays = samples_to_arrays(samples)
        out_path = OUT_DIR / f"{name}.npz"
        np.savez_compressed(out_path, **arrays)
        print(f"저장: {out_path}  x_norm shape={arrays['x_norm'].shape}  y_norm shape={arrays['y_norm'].shape}")


if __name__ == "__main__":
    main()
