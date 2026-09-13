"""Reference DB(manifest.json + sequences.npz) 로드 유틸리티."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .features import extract_all_features
from .reference_levels import (
    DIFFICULTY_LEVELS,
    REFERENCE_CLASSES,
    validate_difficulty_level,
)


_REFERENCE_TIERS = ("ground_truth", "operational")


def _read_manifest(db_dir: Path) -> list[dict]:
    return json.loads((db_dir / "manifest.json").read_text(encoding="utf-8"))["entries"]


def _validate_entry_levels(entries: list[dict]) -> None:
    """난이도 인덱싱 전에 모든 manifest entry의 난이도 metadata를 검증한다."""
    for index, entry in enumerate(entries):
        if "difficulty_level" not in entry:
            identifier = entry.get("medoid_id", entry.get("array_key", f"entry #{index}"))
            raise ValueError(
                "난이도별 Reference DB를 사용하려면 모든 entry에 difficulty_level이 "
                f"필요합니다: {identifier!r}에서 누락 (Reference DB 재구축 필요)"
            )
        try:
            validate_difficulty_level(entry["difficulty_level"])
        except ValueError as error:
            identifier = entry.get("medoid_id", entry.get("array_key", f"entry #{index}"))
            raise ValueError(f"Reference DB entry {identifier!r}: {error}") from error


def _record(entry: dict, arrays: np.lib.npyio.NpzFile) -> dict:
    coords = arrays[entry["array_key"]]
    return {
        "feat": extract_all_features(coords),
        "bounds": entry["phase_boundaries"],
        "meta": entry,
    }


def _validate_required_classes(
    db_by_level: dict[str, dict[str, dict[str, list[dict]]]],
    levels: tuple[str, ...],
) -> None:
    missing = []
    for tier in _REFERENCE_TIERS:
        for level in levels:
            for class_label in REFERENCE_CLASSES:
                if not db_by_level[tier][level][class_label]:
                    missing.append(f"tier={tier}, level={level}, class={class_label}")
    if missing:
        raise ValueError(
            "난이도별 Reference DB에 필수 class 레퍼런스가 없습니다: "
            + "; ".join(missing)
            + " (Reference DB 재구축 필요)"
        )


def load_reference_db(
    db_dir: str | Path,
    difficulty_level: str | None = None,
) -> dict[str, dict[str, list[dict]]]:
    """Reference DB를 기존의 ``db[tier][class_label]`` 형태로 반환한다.

    ``difficulty_level``이 ``None``이면 난이도 metadata를 검사하지 않고 모든
    레퍼런스를 합쳐 반환한다. 이는 ``difficulty_level`` 필드가 없던 구 manifest를
    포함해 기존 호출과 완전히 같은 동작이다. 난이도를 지정하면 해당 난이도만
    반환하며, level-aware DB의 metadata 및 필수 class 완전성을 함께 검증한다.
    """
    db_dir = Path(db_dir)
    entries = _read_manifest(db_dir)

    db: dict[str, dict[str, list[dict]]] = {"ground_truth": defaultdict(list), "operational": defaultdict(list)}
    if difficulty_level is None:
        with np.load(db_dir / "sequences.npz") as arrays:
            for entry in entries:
                db[entry["tier"]][entry["class_label"]].append(_record(entry, arrays))
        return db

    level = validate_difficulty_level(difficulty_level)
    _validate_entry_levels(entries)

    by_level: dict[str, dict[str, dict[str, list[dict]]]] = {
        tier: {level: defaultdict(list)} for tier in _REFERENCE_TIERS
    }
    with np.load(db_dir / "sequences.npz") as arrays:
        for entry in entries:
            if entry["difficulty_level"] != level:
                continue
            record = _record(entry, arrays)
            db[entry["tier"]][entry["class_label"]].append(record)
            by_level[entry["tier"]][level][entry["class_label"]].append(record)
    _validate_required_classes(by_level, (level,))
    return db


def load_reference_db_by_level(
    db_dir: str | Path,
) -> dict[str, dict[str, dict[str, list[dict]]]]:
    """반환: ``db[tier][difficulty_level][class_label] = [reference, ...]``.

    모든 지원 난이도와 필수 자세 class가 두 reference tier에 존재해야 한다.
    누락되거나 알 수 없는 난이도 metadata는 자동 혼합하지 않고 명확히 거부한다.
    """
    db_dir = Path(db_dir)
    entries = _read_manifest(db_dir)
    _validate_entry_levels(entries)

    db: dict[str, dict[str, dict[str, list[dict]]]] = {
        tier: {level: defaultdict(list) for level in DIFFICULTY_LEVELS}
        for tier in _REFERENCE_TIERS
    }
    with np.load(db_dir / "sequences.npz") as arrays:
        for entry in entries:
            level = entry["difficulty_level"]
            db[entry["tier"]][level][entry["class_label"]].append(_record(entry, arrays))

    _validate_required_classes(db, DIFFICULTY_LEVELS)
    return db


__all__ = ["load_reference_db", "load_reference_db_by_level"]
