#!/usr/bin/env python3
"""설정된 에어스쿼트 데이터셋을 actor 단위 Train/Validation으로 분할한다."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trainer.actor_split import (  # noqa: E402
    build_actor_split,
    check_no_leakage,
    load_air_squat_sequences,
    save_split,
    summarize_split,
)
from ai_trainer.dataset_config import DATASET_PATH  # noqa: E402

OUT_PATH = Path(__file__).resolve().parent.parent / "configs" / "actor_split.json"


def main() -> None:
    sequences = load_air_squat_sequences(DATASET_PATH)
    print(f"데이터셋: {DATASET_PATH}")
    print(f"전체 에어스쿼트 시퀀스: {len(sequences)}개")

    split = build_actor_split(sequences, val_ratio=0.2, seed=42)
    problems = check_no_leakage(sequences, split)
    print(f"\n[재분할 leakage 검사] 문제 {len(problems)}건")
    for p in problems:
        print("  -", p)

    summary = summarize_split(sequences, split)
    print("\n[재분할 요약]")
    print(f"  actor 수: train={summary['n_actors']['train']}, val={summary['n_actors']['val']}")
    for side in ["train", "val"]:
        row = summary["n_sequences"][side]
        print(
            f"  {side:5s}: 정상={row['정상']:3d}  발뒤꿈치오류={row['발뒤꿈치오류']:3d}  "
            f"엉덩이하방오류={row['엉덩이하방오류']:3d}  고관절오류={row['고관절오류']:3d}  "
            f"합계={row['TOTAL']:3d}"
        )

    save_split(split, sequences, OUT_PATH, val_ratio=0.2, seed=42)
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
