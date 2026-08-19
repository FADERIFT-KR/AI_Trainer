#!/usr/bin/env python3
"""정상/오류 4개 클래스에 대해 phase feature(pelvis height/velocity, knee/hip flexion)를
여러 시퀀스에 걸쳐 시각화한다 — phase segmentation 기준을 정하기 전에 먼저
feature가 클래스 내에서 실제로 안정적인 패턴을 보이는지 눈으로 확인하기 위함.

Ground Truth Reference(3d_points.csv) 기준으로 그린다 (오류유형별 특징 분석 목적).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402

from ai_trainer.actor_split import load_all_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference  # noqa: E402

TL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Training/02.라벨링데이터/TL.zip"
)
VL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "output" / "phase_feature_check.png"

CLASSES = ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"]
N_PER_CLASS = 10
RESAMPLE_N = 100  # 정규화 시간축(0~100%) 오버레이용


def resample(x: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    t_src = np.linspace(0, 1, len(x))
    t_dst = np.linspace(0, 1, n)
    return np.interp(t_dst, t_src, x)


def main() -> None:
    import random

    rng = random.Random(3)
    all_seqs = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    by_class = {c: [] for c in CLASSES}
    for os_ in all_seqs:
        if os_.seq.error_type in by_class:
            by_class[os_.seq.error_type].append(os_)
    for c in CLASSES:
        rng.shuffle(by_class[c])

    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}

    fig, axes = plt.subplots(4, 4, figsize=(20, 14), sharex=True)
    feature_names = ["pelvis_height", "pelvis_velocity", "knee_flexion_deg", "hip_flexion_deg"]

    for col, cls in enumerate(CLASSES):
        curves = {f: [] for f in feature_names}
        n_done = 0
        for os_ in by_class[cls]:
            if n_done >= N_PER_CLASS:
                break
            ref = build_ground_truth_reference(zips[os_.origin], os_.seq, os_.origin)
            if ref is None:
                continue
            feats = extract_phase_features(ref.coords)
            for f in feature_names:
                curves[f].append(resample(getattr(feats, f)))
            n_done += 1

        t_axis = np.linspace(0, 100, RESAMPLE_N)
        for row, f in enumerate(feature_names):
            ax = axes[row, col]
            arr = np.stack(curves[f])  # (n_done, RESAMPLE_N)
            for line in arr:
                ax.plot(t_axis, line, color="tab:blue", alpha=0.25, linewidth=1)
            ax.plot(t_axis, arr.mean(axis=0), color="tab:red", linewidth=2, label="mean")
            if row == 0:
                ax.set_title(f"{cls} (n={n_done})")
            if col == 0:
                ax.set_ylabel(f)
            if row == 3:
                ax.set_xlabel("동작 진행률 (%)")
            ax.grid(alpha=0.3)

    for z in zips.values():
        z.close()

    fig.suptitle("클래스별 Phase Feature (Ground Truth Reference, 시퀀스 진행률 0~100% 정규화 오버레이)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=110)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
