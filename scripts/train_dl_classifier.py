#!/usr/bin/env python3
"""SmallSquatCNN(ai_trainer/dl_classifier.py)을 학습해 output/dl_classifier/에 저장한다.

actor_split.json의 train actor로만 학습하고 val actor로 평가한다(다른 모든 산출물과
동일한 leakage-방지 원칙). 2026-09-08 actor 단위 5-fold 교차검증에서 이미
AP 95.8%±1.5%(DTW 88.8%±7.4% 대비)로 검증된 구성을 그대로 이 단일 split에 적용한다
— 여기서 나오는 val 숫자는 "실제로 배포될 모델"의 참고용 성능이고, 진짜 신뢰 구간은
위 교차검증 결과를 봐야 한다(단일 split은 val actor가 적어(14~27명) 그 자체로도
fold마다 74~98%로 흔들렸던 걸 오늘 이미 확인했다).

출력
----
  output/dl_classifier/model.pt        state_dict
  output/dl_classifier/norm_stats.npz  mu, sigma (train set 기준 z-score 정규화)
  output/dl_classifier/train_report.json  val 성능 기록
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from ai_trainer.actor_split import load_all_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.dl_classifier import CLASSES, PHASES, SmallSquatCNN, build_fixed_vector  # noqa: E402
from ai_trainer.features import extract_all_features  # noqa: E402
from ai_trainer.lifting_dataset import load_actor_split  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.phase_segmentation import segment_phases  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference  # noqa: E402

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
OUT_DIR = ROOT / "output" / "dl_classifier"
SEED = 0
EPOCHS = 40
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-3


def average_precision(records: list[tuple[float, bool]]) -> float:
    records = sorted(records, key=lambda r: -r[0])
    n_pos = sum(1 for _, y in records if y)
    if n_pos == 0:
        return 0.0
    hits, precisions = 0, []
    for i, (_, y) in enumerate(records, start=1):
        if y:
            hits += 1
            precisions.append(hits / i)
    return sum(precisions) / n_pos


def build_dataset(seqs, zips) -> list[dict]:
    out = []
    for os_ in seqs:
        ref = build_ground_truth_reference(zips[os_.origin], os_.seq, os_.origin)
        if ref is None or ref.coords.shape[0] < 10:
            continue
        pf = extract_phase_features(ref.coords)
        bounds = segment_phases(pf).as_dict()
        feat = extract_all_features(ref.coords)
        vec = build_fixed_vector(feat, bounds)
        if vec is None:
            continue
        out.append({"vec": vec, "cls": os_.seq.error_type})
    return out


def main() -> None:
    torch.manual_seed(SEED)
    actor_to_split = load_actor_split(SPLIT_PATH)
    all_seqs = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    train_seqs = [os_ for os_ in all_seqs if actor_to_split.get(os_.seq.actor) == "train"]
    val_seqs = [os_ for os_ in all_seqs if actor_to_split.get(os_.seq.actor) == "val"]
    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}

    print("=== 데이터셋 빌드 ===")
    t0 = time.time()
    train_data = build_dataset(train_seqs, zips)
    val_data = build_dataset(val_seqs, zips)
    for z in zips.values():
        z.close()
    print(f"train {len(train_data)}개, val {len(val_data)}개 ({time.time()-t0:.1f}s)")
    cls_idx = {c: i for i, c in enumerate(CLASSES)}
    for c in CLASSES:
        print(f"  {c}: train {sum(1 for d in train_data if d['cls']==c)} / val {sum(1 for d in val_data if d['cls']==c)}")

    Xtr = np.stack([d["vec"] for d in train_data]).transpose(0, 2, 1)  # (N, in_ch, T)
    ytr = np.array([cls_idx[d["cls"]] for d in train_data])
    mu = Xtr.mean(axis=(0, 2), keepdims=True)
    sigma = Xtr.std(axis=(0, 2), keepdims=True) + 1e-6
    Xtr_n = (Xtr - mu) / sigma

    Xva = np.stack([d["vec"] for d in val_data]).transpose(0, 2, 1)
    yva = np.array([cls_idx[d["cls"]] for d in val_data])
    Xva_n = (Xva - mu) / sigma

    counts = np.bincount(ytr, minlength=len(CLASSES)).astype(np.float32)
    class_weight = torch.tensor(counts.sum() / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)

    print("\n=== 학습 ===")
    model = SmallSquatCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)

    Xtr_t = torch.tensor(Xtr_n, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.long)
    model.train()
    n = Xtr_t.shape[0]
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            opt.zero_grad()
            out = model(Xtr_t[idx])
            loss = loss_fn(out, ytr_t[idx])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{EPOCHS}: loss={epoch_loss/n:.4f}")

    print("\n=== val 평가 ===")
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(Xva_n, dtype=torch.float32)), dim=1).numpy()
    pred = probs.argmax(axis=1)
    acc = float((pred == yva).mean()) * 100
    records = [(probs[i, cls_idx["정상"]], yva[i] == cls_idx["정상"]) for i in range(len(yva))]
    ap = average_precision(records) * 100
    print(f"val 4클래스 정확도: {acc:.1f}%")
    print(f"val AP(정상 vs 오류): {ap:.1f}%")
    print("(참고: 2026-09-08 actor 5-fold 교차검증 기준 DL AP는 95.8%±1.5% — 이 단일")
    print(" split의 숫자는 val actor 수가 적어 그보다 위아래로 흔들릴 수 있음)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUT_DIR / "model.pt")
    np.savez(OUT_DIR / "norm_stats.npz", mu=mu, sigma=sigma)
    (OUT_DIR / "train_report.json").write_text(
        json.dumps(
            {
                "val_accuracy_4class": acc,
                "val_ap_normal_vs_error": ap,
                "n_train": len(train_data),
                "n_val": len(val_data),
                "crossval_reference": "actor 5-fold CV(2026-09-08): DL AP 95.8%+-1.5%, DTW AP 88.8%+-7.4%",
                "hyperparams": {"epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "seed": SEED},
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n저장 완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
