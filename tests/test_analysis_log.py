import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

from ai_trainer.game_ui.analysis_log import SessionAnalysisLog


def _rep(index=0, predicted_class="정상", score=90.0):
    return SimpleNamespace(
        rep_index=index,
        predicted_class=predicted_class,
        score_vs_normal=score,
        raw_distance_by_class={predicted_class: 0.12},
        top_contributing_features=[("joint_coords_3d", 0.3)],
        posture_score=None,
    )


class SessionAnalysisLogTests(unittest.TestCase):
    def test_one_completed_rep_increases_entries_by_exactly_one(self):
        history = SessionAnalysisLog()
        before = len(history.entries)
        history.record_completed_rep("에어스쿼트", _rep(index=0))
        self.assertEqual(len(history.entries), before + 1)

    def test_normal_completed_rep_is_stored(self):
        history = SessionAnalysisLog()
        history.record_completed_rep("에어스쿼트", _rep(predicted_class="정상"))
        self.assertEqual(history.entries[0].status, "normal")

    def test_error_completed_rep_is_stored(self):
        history = SessionAnalysisLog()
        history.record_completed_rep("에어스쿼트", _rep(predicted_class="고관절오류"))
        self.assertEqual(history.entries[0].status, "error")

    def test_partial_is_excluded_from_final_rep_statistics(self):
        history = SessionAnalysisLog()
        partial = {"distance_by_class": {"정상": 0.2, "고관절오류": 0.1}}
        for _ in range(history.PARTIAL_STABLE_FRAMES):
            history.observe_partial("에어스쿼트", partial)
        summary = history.summary("에어스쿼트")
        self.assertEqual(summary.total_reps, 0)
        self.assertEqual(summary.normal_count, 0)
        self.assertEqual(summary.error_count, 0)

    def test_duplicate_completed_rep_index_is_not_stored_twice(self):
        history = SessionAnalysisLog()
        history.record_completed_rep("에어스쿼트", _rep(index=0))
        history.record_completed_rep("에어스쿼트", _rep(index=0))
        self.assertEqual(len(history.entries), 1)

    def test_records_one_structured_entry_per_completed_rep(self):
        history = SessionAnalysisLog()

        entry = history.record_completed_rep("에어스쿼트", _rep())
        duplicate = history.record_completed_rep("에어스쿼트", _rep())

        self.assertEqual(entry.status, "normal")
        self.assertEqual(entry.rep, 1)
        self.assertEqual(entry.source, "rep")
        self.assertEqual(entry.dtw_distance, 0.12)
        self.assertIsNone(duplicate)
        self.assertEqual(len(history.entries), 1)

    def test_error_uses_existing_explanation_and_summary_counts(self):
        history = SessionAnalysisLog()
        history.record_completed_rep("에어스쿼트", _rep(predicted_class="발뒤꿈치오류", score=None))

        entry = history.entries[0]
        summary = history.summary("에어스쿼트")

        self.assertEqual(entry.error_messages, ("발뒤꿈치가 들려요",))
        self.assertEqual(entry.body_parts, ("발뒤꿈치",))
        self.assertEqual(summary.error_count, 1)
        self.assertEqual(summary.total_records, 1)
        self.assertEqual(summary.total_reps, 1)
        self.assertEqual(summary.error_counts["발뒤꿈치오류"], 1)
        self.assertIsNone(summary.average_score)

    def test_clear_resets_entries_and_duplicate_guard(self):
        history = SessionAnalysisLog()
        history.record_completed_rep("에어스쿼트", _rep())
        history.clear()

        self.assertEqual(history.entries, [])
        self.assertIsNotNone(history.record_completed_rep("에어스쿼트", _rep()))

    def test_completed_final_posture_match_is_stored(self):
        history = SessionAnalysisLog()
        result = _rep()
        result.posture_score = {
            "score_valid": True, "overall": 86.5, "depth": 90.0, "hip": 82.0,
            "knee": 88.0, "heel_stability": 94.0, "balance": 91.0,
            "trajectory": 78.0,
        }
        entry = history.record_completed_rep("에어스쿼트", result)
        self.assertEqual(entry.score, 86.5)
        self.assertEqual(entry.posture_components["heel_stability"], 94.0)

    def test_stable_partial_is_recorded_without_rep_end_and_is_debounced(self):
        history = SessionAnalysisLog()
        partial = {"distance_by_class": {"정상": 0.2, "고관절오류": 0.1}}
        started = datetime(2026, 9, 1, 14, 30, 0)

        recorded = None
        for _ in range(history.PARTIAL_STABLE_FRAMES):
            recorded = history.observe_partial("에어스쿼트", partial, now=started)

        self.assertIsNotNone(recorded)
        self.assertIsNone(recorded.rep)
        self.assertEqual(recorded.source, "partial")
        self.assertEqual(recorded.predicted_class, "고관절오류")
        self.assertIsNone(
            history.observe_partial("에어스쿼트", partial, now=started + timedelta(seconds=5))
        )
        self.assertIsNotNone(
            history.observe_partial("에어스쿼트", partial, now=started + timedelta(seconds=11))
        )
        summary = history.summary("에어스쿼트")
        self.assertEqual(summary.total_records, 2)
        self.assertEqual(summary.total_reps, 0)


if __name__ == "__main__":
    unittest.main()
