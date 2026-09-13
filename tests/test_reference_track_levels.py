"""게임 UI 레퍼런스 트랙의 난이도 선택 계약 테스트."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.game_ui import reference_track
from ai_trainer.reference_levels import DIFFICULTY_LEVELS


def _entry(
    level: str,
    marker: int,
    *,
    class_label: str = "정상",
    rank: int = 0,
    tier: str = "ground_truth",
    explicit_rank: bool = True,
    id_rank: int | None = None,
) -> dict:
    entry = {
        "medoid_id": f"{class_label}_{rank if id_rank is None else id_rank}_actor_rep1",
        "class_label": class_label,
        "difficulty_level": level,
        "tier": tier,
        "array_key": f"coords_{marker}",
        "phase_boundaries": {"준비": [0, 2]},
        "marker": marker,
    }
    if explicit_rank:
        entry["medoid_rank"] = rank
    return entry


class SyntheticReferenceDb:
    def __init__(self, entries: list[dict]):
        self._temporary_dir = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_dir.name)
        self.write(entries)

    def write(self, entries: list[dict]) -> None:
        (self.path / "manifest.json").write_text(
            json.dumps({"entries": entries}, ensure_ascii=False),
            encoding="utf-8",
        )
        arrays = {
            entry["array_key"]: np.full(
                (2, 18, 3), entry["marker"], dtype=np.float32
            )
            for entry in entries
        }
        np.savez_compressed(self.path / "sequences.npz", **arrays)

    def cleanup(self) -> None:
        self._temporary_dir.cleanup()


class ReferenceTrackLevelTests(unittest.TestCase):
    def setUp(self) -> None:
        # 의도적으로 난이도 순서를 뒤섞고 medoid_id의 rank를 틀리게 둔다.
        # v2 medoid_rank와 명시적 difficulty_level로 선택해야만 통과한다.
        self.entries = [
            _entry("고급", 30, id_rank=99),
            _entry("초급", 10, id_rank=99),
            _entry("중급", 20, id_rank=99),
        ]
        self.db = SyntheticReferenceDb(self.entries)
        self.db_patch = patch.object(reference_track, "DB_DIR", self.db.path)
        self.db_patch.start()

    def tearDown(self) -> None:
        self.db_patch.stop()
        self.db.cleanup()

    def test_explicit_level_selection_is_independent_of_manifest_order(self) -> None:
        expected_marker = {"초급": 10, "중급": 20, "고급": 30}

        for level in DIFFICULTY_LEVELS:
            with self.subTest(level=level):
                track = reference_track.ReferenceTrack(difficulty_level=level)
                self.assertEqual(track.meta["difficulty_level"], level)
                self.assertEqual(track.meta["medoid_rank"], 0)
                np.testing.assert_array_equal(
                    track.coords,
                    np.full((2, 18, 3), expected_marker[level], dtype=np.float32),
                )

    def test_create_normal_tracks_returns_all_three_levels(self) -> None:
        tracks = reference_track.create_normal_tracks()

        self.assertEqual(tuple(tracks), DIFFICULTY_LEVELS)
        for level, track in tracks.items():
            self.assertEqual(track.meta["class_label"], "정상")
            self.assertEqual(track.meta["difficulty_level"], level)
            self.assertEqual(track.meta["medoid_rank"], 0)

    def test_missing_normal_level_is_reported(self) -> None:
        self.db.write(
            [entry for entry in self.entries if entry["difficulty_level"] != "중급"]
        )

        with self.assertRaisesRegex(ValueError, "중급"):
            reference_track.create_normal_tracks()

    def test_legacy_medoid_id_rank_fallback(self) -> None:
        legacy = _entry("초급", 7, explicit_rank=False)
        self.db.write([legacy])

        track = reference_track.ReferenceTrack(difficulty_level="초급")

        self.assertEqual(track.meta["medoid_id"], "정상_0_actor_rep1")
        self.assertEqual(
            reference_track.list_available(difficulty_level="초급"),
            [("정상", 0)],
        )

    def test_list_available_keeps_tuple_contract_and_filters_level(self) -> None:
        extra = _entry("중급", 40, class_label="고관절오류", rank=1)
        self.db.write([extra, *self.entries])

        self.assertEqual(
            reference_track.list_available(difficulty_level="초급"),
            [("정상", 0)],
        )
        self.assertEqual(
            reference_track.list_available(difficulty_level="중급"),
            [("고관절오류", 1), ("정상", 0)],
        )
        self.assertEqual(
            reference_track.list_available(),
            [("고관절오류", 1), ("정상", 0)],
        )

    def test_unknown_level_is_rejected_before_lookup(self) -> None:
        with self.assertRaisesRegex(ValueError, "알 수 없는 난이도"):
            reference_track.ReferenceTrack(difficulty_level="전문가")
        with self.assertRaisesRegex(ValueError, "알 수 없는 난이도"):
            reference_track.list_available(difficulty_level="전문가")


if __name__ == "__main__":
    unittest.main()
