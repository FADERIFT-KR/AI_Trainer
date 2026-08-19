#!/usr/bin/env python3
"""actor(피험자) 단위 Train/Validation 분할을 검증하고 재구성한다.

1. AI Hub 공식 TL.zip/VL.zip 분할이 실제로 피험자 단위인지 검사한다.
2. leakage가 있으면 TL+VL을 합쳐 actor 단위로 새 분할을 만들고
   configs/actor_split.json에 저장한다.
3. 새 분할에 대해 다시 leakage 검사를 수행해 통과하는지 확인한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai_trainer.actor_split import (  # noqa: E402
    build_actor_split,
    check_no_leakage,
    load_all_air_squat_sequences,
    save_split,
    summarize_split,
)

TL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Training/02.라벨링데이터/TL.zip"
)
VL_ZIP = (
    "/Users/faderift/Project/Crossfit_Labeling_Data/213.크로스핏_동작_데이터/"
    "01-1.정식개방데이터/Validation/02.라벨링데이터/VL.zip"
)
OUT_PATH = Path(__file__).resolve().parent.parent / "configs" / "actor_split.json"


def main() -> None:
    sequences = load_all_air_squat_sequences(TL_ZIP, VL_ZIP)
    print(f"전체 에어스쿼트 시퀀스(TL+VL 합계): {len(sequences)}개")

    # 1) AI Hub 공식 분할(origin=TL/VL 그대로)을 "split"으로 취급해 leakage 검사
    official_split = {os_.seq.actor: ("train" if os_.origin == "TL" else "val") for os_ in sequences}
    # 주의: 한 actor가 TL/VL 양쪽에 다 있으면 dict comprehension에서 마지막 값(VL)으로 덮어써진다.
    # 실제 겹치는지 여부는 원본 (origin, actor) 쌍으로 별도 확인한다.
    actor_origins: dict[str, set[str]] = {}
    for os_ in sequences:
        actor_origins.setdefault(os_.seq.actor, set()).add(os_.origin)
    overlapping = {a: o for a, o in actor_origins.items() if len(o) > 1}
    print(f"\n[공식 TL/VL 분할 검사] TL과 VL에 동시에 등장하는 actor 수: {len(overlapping)} / {len(actor_origins)}")
    if overlapping:
        print("  => 공식 분할은 피험자 단위가 아님 (leakage 존재). actor 단위로 재분할한다.")

    # 2) actor 단위 재분할 생성
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
