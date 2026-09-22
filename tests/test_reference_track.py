"""Regression tests for reference medoid manifest parsing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ai_trainer.squat.game_ui.reference_track import medoid_rank_from_entry  # noqa: E402


class MedoidRankTests(unittest.TestCase):
    def test_current_manifest_uses_explicit_rank_despite_difficulty_in_id(self) -> None:
        entry = {"medoid_id": "normal_beginner_3_CB14_rep4", "medoid_rank": 3}
        self.assertEqual(medoid_rank_from_entry(entry), 3)

    def test_legacy_manifest_uses_the_numeric_id_component(self) -> None:
        entry = {"medoid_id": "normal_2_CA01_rep1"}
        self.assertEqual(medoid_rank_from_entry(entry), 2)

    def test_missing_rank_is_reported_clearly(self) -> None:
        with self.assertRaisesRegex(ValueError, "medoid rank"):
            medoid_rank_from_entry({"medoid_id": "normal_beginner_CB14_rep4"})


if __name__ == "__main__":
    unittest.main()
