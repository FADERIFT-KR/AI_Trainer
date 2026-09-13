"""난이도별 Reference DB 로드 계약 테스트."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.reference_db_io import load_reference_db, load_reference_db_by_level
from ai_trainer.reference_levels import DIFFICULTY_LEVELS, REFERENCE_CLASSES


REFERENCE_TIERS = ("ground_truth", "operational")


def _complete_entries() -> list[dict]:
    entries = []
    marker = 1
    for tier in REFERENCE_TIERS:
        for level in DIFFICULTY_LEVELS:
            for class_label in REFERENCE_CLASSES:
                array_key = f"reference_{marker}"
                entries.append(
                    {
                        "medoid_id": f"{class_label}_{level}_{marker}",
                        "class_label": class_label,
                        "difficulty_level": level,
                        "tier": tier,
                        "array_key": array_key,
                        "phase_boundaries": {"준비": [0, 2]},
                        "marker": marker,
                    }
                )
                marker += 1
    return entries


class SyntheticReferenceDb:
    def __init__(self, entries: list[dict]):
        self._temporary_dir = tempfile.TemporaryDirectory()
        self.path = Path(self._temporary_dir.name)
        (self.path / "manifest.json").write_text(
            json.dumps(
                {"classes": list(REFERENCE_CLASSES), "entries": entries},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        arrays = {
            entry["array_key"]: np.full((2, 18, 3), entry["marker"], dtype=np.float32)
            for entry in entries
        }
        np.savez_compressed(self.path / "sequences.npz", **arrays)

    def cleanup(self) -> None:
        self._temporary_dir.cleanup()


def _fake_features(coords: np.ndarray) -> dict[str, np.ndarray]:
    return {"marker": coords[:, 0, 0].copy()}


class ReferenceDbLevelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = _complete_entries()
        self.db = SyntheticReferenceDb(self.entries)
        self.features_patch = patch(
            "ai_trainer.reference_db_io.extract_all_features",
            side_effect=_fake_features,
        )
        self.extract_features = self.features_patch.start()

    def tearDown(self) -> None:
        self.features_patch.stop()
        self.db.cleanup()

    def _rewrite_manifest(self, entries: list[dict]) -> None:
        (self.db.path / "manifest.json").write_text(
            json.dumps(
                {"classes": list(REFERENCE_CLASSES), "entries": entries},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def test_unfiltered_load_preserves_flat_legacy_shape(self) -> None:
        loaded = load_reference_db(self.db.path)

        self.assertEqual(set(loaded), set(REFERENCE_TIERS))
        for tier in REFERENCE_TIERS:
            self.assertIsInstance(loaded[tier], defaultdict)
            for class_label in REFERENCE_CLASSES:
                references = loaded[tier][class_label]
                self.assertEqual(len(references), len(DIFFICULTY_LEVELS))
                self.assertEqual(
                    [reference["meta"]["difficulty_level"] for reference in references],
                    list(DIFFICULTY_LEVELS),
                )

    def test_filtered_load_returns_only_the_requested_level(self) -> None:
        loaded = load_reference_db(self.db.path, difficulty_level="중급")

        for tier in REFERENCE_TIERS:
            for class_label in REFERENCE_CLASSES:
                references = loaded[tier][class_label]
                self.assertEqual(len(references), 1)
                self.assertEqual(references[0]["meta"]["difficulty_level"], "중급")

    def test_load_by_level_builds_tier_level_class_index(self) -> None:
        loaded = load_reference_db_by_level(self.db.path)

        self.assertEqual(set(loaded), set(REFERENCE_TIERS))
        for tier in REFERENCE_TIERS:
            self.assertEqual(tuple(loaded[tier]), DIFFICULTY_LEVELS)
            for level in DIFFICULTY_LEVELS:
                self.assertIsInstance(loaded[tier][level], defaultdict)
                for class_label in REFERENCE_CLASSES:
                    references = loaded[tier][level][class_label]
                    self.assertEqual(len(references), 1)
                    self.assertEqual(references[0]["meta"]["difficulty_level"], level)
                    np.testing.assert_array_equal(
                        references[0]["feat"]["marker"],
                        np.full(2, references[0]["meta"]["marker"], dtype=np.float32),
                    )

    def test_unknown_requested_level_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "알 수 없는 난이도"):
            load_reference_db(self.db.path, difficulty_level="전문가")

    def test_legacy_manifest_without_level_metadata_still_loads_unfiltered(self) -> None:
        legacy_entries = []
        for entry in self.entries:
            legacy_entry = dict(entry)
            legacy_entry.pop("difficulty_level")
            legacy_entries.append(legacy_entry)
        self._rewrite_manifest(legacy_entries)

        loaded = load_reference_db(self.db.path)

        for tier in REFERENCE_TIERS:
            for class_label in REFERENCE_CLASSES:
                self.assertEqual(len(loaded[tier][class_label]), len(DIFFICULTY_LEVELS))

    def test_level_aware_loads_reject_missing_level_metadata(self) -> None:
        entries = [dict(entry) for entry in self.entries]
        entries[0].pop("difficulty_level")
        self._rewrite_manifest(entries)

        for loader in (
            lambda: load_reference_db(self.db.path, difficulty_level="초급"),
            lambda: load_reference_db_by_level(self.db.path),
        ):
            with self.subTest(loader=loader), self.assertRaisesRegex(
                ValueError, "difficulty_level.*누락"
            ):
                loader()

    def test_level_aware_loads_reject_unknown_entry_level(self) -> None:
        entries = [dict(entry) for entry in self.entries]
        entries[0]["difficulty_level"] = "전문가"
        self._rewrite_manifest(entries)

        for loader in (
            lambda: load_reference_db(self.db.path, difficulty_level="초급"),
            lambda: load_reference_db_by_level(self.db.path),
        ):
            with self.subTest(loader=loader), self.assertRaisesRegex(
                ValueError, "알 수 없는 난이도"
            ):
                loader()

    def test_filtered_load_rejects_empty_required_class(self) -> None:
        entries = [
            entry
            for entry in self.entries
            if not (
                entry["tier"] == "operational"
                and entry["difficulty_level"] == "고급"
                and entry["class_label"] == "정상"
            )
        ]
        self._rewrite_manifest(entries)

        with self.assertRaisesRegex(
            ValueError, "tier=operational, level=고급, class=정상"
        ):
            load_reference_db(self.db.path, difficulty_level="고급")

        # 선택하지 않은 난이도의 결손은 해당 난이도만 요청한 flat load를 막지 않는다.
        loaded = load_reference_db(self.db.path, difficulty_level="중급")
        self.assertEqual(len(loaded["operational"]["정상"]), 1)

    def test_load_by_level_rejects_any_empty_required_class(self) -> None:
        entries = [
            entry
            for entry in self.entries
            if not (
                entry["tier"] == "ground_truth"
                and entry["difficulty_level"] == "초급"
                and entry["class_label"] == "고관절오류"
            )
        ]
        self._rewrite_manifest(entries)

        with self.assertRaisesRegex(
            ValueError, "tier=ground_truth, level=초급, class=고관절오류"
        ):
            load_reference_db_by_level(self.db.path)


if __name__ == "__main__":
    unittest.main()
