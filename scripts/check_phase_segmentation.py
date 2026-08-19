#!/usr/bin/env python3
"""phase_segmentation 결과를 실제 시퀀스 몇 개에 대해 pelvis_height 곡선 위에
색칠해서 눈으로 검증한다 (준비/하강/최저점/상승/종료)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False

from ai_trainer.actor_split import load_all_air_squat_sequences  # noqa: E402
from ai_trainer.aihub_zip import AiHubZip  # noqa: E402
from ai_trainer.phase_features import extract_phase_features  # noqa: E402
from ai_trainer.phase_segmentation import PHASES, segment_phases  # noqa: E402
from ai_trainer.reference_pipeline import build_ground_truth_reference  # noqa: E402

TL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Training/02.라벨링데이터/TL.zip"
)
VL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "output" / "phase_segmentation_check.png"

CLASSES = ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류"]
PHASE_COLORS = {"준비": "#999999", "하강": "#4C72B0", "최저점": "#C44E52", "상승": "#55A868", "종료": "#999999"}


def main() -> None:
    rng = random.Random(11)
    all_seqs = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    by_class = {c: [] for c in CLASSES}
    for os_ in all_seqs:
        if os_.seq.error_type in by_class:
            by_class[os_.seq.error_type].append(os_)
    for c in CLASSES:
        rng.shuffle(by_class[c])

    zips = {"TL": AiHubZip(TL_ZIP), "VL": AiHubZip(VL_ZIP)}

    n_rows, n_cols = 4, 3  # 클래스당 3개 샘플
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12))

    for row, cls in enumerate(CLASSES):
        shown = 0
        for os_ in by_class[cls]:
            if shown >= n_cols:
                break
            ref = build_ground_truth_reference(zips[os_.origin], os_.seq, os_.origin)
            if ref is None:
                continue
            feats = extract_phase_features(ref.coords)
            bounds = segment_phases(feats)

            ax = axes[row, shown]
            t = range(len(feats.pelvis_height))
            for phase, (s, e) in bounds.as_dict().items():
                if e > s:
                    ax.axvspan(s, e, color=PHASE_COLORS[phase], alpha=0.35, label=phase)
            ax.plot(t, feats.pelvis_height, color="black", linewidth=1.5)
            ax.set_title(f"{cls} {os_.seq.actor} rep{os_.seq.rep}", fontsize=10)
            ax.set_ylim(0, 1.2)
            if row == 0 and shown == 0:
                handles = [plt.Rectangle((0, 0), 1, 1, color=PHASE_COLORS[p], alpha=0.35) for p in PHASES]
                ax.legend(handles, PHASES, loc="upper right", fontsize=7)
            shown += 1

    for z in zips.values():
        z.close()

    fig.suptitle("Phase segmentation 검증 (검은 선 = pelvis_height, 배경색 = 판별된 phase)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=110)
    print(f"저장: {OUT_PATH}")


if __name__ == "__main__":
    main()
