"""actor(피험자) 단위 Train/Validation 재분할.

AI Hub 공식 TL.zip/VL.zip 분할은 피험자 단위가 아니다(Validation의 actor 26명이
전부 Training에도 등장). 이 모듈은 TL+VL을 합친 뒤 **actor를 원자 단위**로 삼아
난이도(actor 접두어 CA=고급/CB=초급/CI=중급) 비율을 유지하며 새로 분할한다.

actor가 분할 단위이므로, 동일 actor의 서로 다른 repetition/camera/난이도 데이터가
train/val에 동시에 들어가는 leakage는 이 방식으로 구조적으로 발생할 수 없다
(난이도는 actor 접두어로 고정되어 한 actor가 여러 난이도에 걸치지 않는다).
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .aihub_zip import AiHubZip, SequenceKey

_LEVEL_PREFIX = {"CA": "고급", "CB": "초급", "CI": "중급"}


@dataclass(frozen=True)
class OriginSequence:
    """(원본 zip 출처, 시퀀스 키)."""

    origin: str  # "TL" or "VL"
    seq: SequenceKey


def load_all_air_squat_sequences(tl_zip: str | Path, vl_zip: str | Path) -> list[OriginSequence]:
    """TL.zip과 VL.zip을 합쳐 에어스쿼트 전체 시퀀스 목록을 만든다."""
    out: list[OriginSequence] = []
    for origin, path in (("TL", tl_zip), ("VL", vl_zip)):
        with AiHubZip(path) as z:
            for seq in z.iter_air_squat_sequences():
                out.append(OriginSequence(origin, seq))
    return out


def build_actor_split(
    sequences: list[OriginSequence],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, str]:
    """actor -> "train"/"val" 매핑을 만든다. 난이도(접두어)별 비율을 유지한다.

    actor가 최소 분할 단위이므로 결과적으로 동일 actor의 모든 시퀀스(모든
    repetition/camera/오류유형)가 항상 같은 쪽에만 배정된다.
    """
    actors_by_prefix: dict[str, set[str]] = defaultdict(set)
    for os_ in sequences:
        actors_by_prefix[os_.seq.actor[:2]].add(os_.seq.actor)

    rng = random.Random(seed)
    split: dict[str, str] = {}
    for prefix, actors in actors_by_prefix.items():
        actors = sorted(actors)  # 결정론적 순서 고정
        rng.shuffle(actors)
        n_val = max(1, round(len(actors) * val_ratio))
        val_actors = set(actors[:n_val])
        for a in actors:
            split[a] = "val" if a in val_actors else "train"
    return split


def summarize_split(sequences: list[OriginSequence], split: dict[str, str]) -> dict:
    """분할 결과를 오류유형/난이도별로 집계한다 (보고 및 검증용)."""
    counts: Counter = Counter()
    actors_per_side: dict[str, set[str]] = {"train": set(), "val": set()}
    for os_ in sequences:
        side = split[os_.seq.actor]
        counts[(side, os_.seq.error_type)] += 1
        counts[(side, "TOTAL")] += 1
        actors_per_side[side].add(os_.seq.actor)

    return {
        "n_actors": {side: len(a) for side, a in actors_per_side.items()},
        "n_sequences": {
            side: {
                err: counts[(side, err)]
                for err in ["정상", "발뒤꿈치오류", "엉덩이하방오류", "고관절오류", "TOTAL"]
            }
            for side in ["train", "val"]
        },
    }


def check_no_leakage(sequences: list[OriginSequence], split: dict[str, str]) -> list[str]:
    """동일 actor가 train과 val에 동시에 존재하는지 검사한다. 문제 목록(빈 리스트면 정상)을 반환."""
    problems = []
    sides_by_actor: dict[str, set[str]] = defaultdict(set)
    for os_ in sequences:
        sides_by_actor[os_.seq.actor].add(split[os_.seq.actor])
    for actor, sides in sides_by_actor.items():
        if len(sides) > 1:
            problems.append(f"actor {actor}가 여러 split에 동시에 존재함: {sides}")
    return problems


def save_split(
    split: dict[str, str],
    sequences: list[OriginSequence],
    out_path: str | Path,
    val_ratio: float,
    seed: int,
) -> None:
    summary = summarize_split(sequences, split)
    payload = {
        "description": (
            "actor(피험자) 단위 재분할. AI Hub 공식 TL/VL 분할은 피험자 단위가 아니므로 "
            "(VL의 모든 actor가 TL에도 존재) TL+VL을 합친 뒤 actor 단위로 새로 나눴다."
        ),
        "val_ratio": val_ratio,
        "seed": seed,
        "actor_to_split": dict(sorted(split.items())),
        "summary": summary,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
