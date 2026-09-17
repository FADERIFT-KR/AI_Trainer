import time
import unittest

import numpy as np

from ai_trainer.online_dtw import OnlineSquatSession, RepResult


class StubFinalAnalysisSession(OnlineSquatSession):
    def _finalize_rep_sync(self, end_t):
        # Deliberately slower than the submit path so non-blocking behavior is observable.
        time.sleep(0.04)
        rep_index = len(self.completed_reps)
        marker = float(self.raw2d_buffer[0][0, 0])
        self.completed_reps.append(
            RepResult(rep_index, (0, end_t), f"class-{marker}", {"normal": marker},
                      None, [], posture_score={"score_valid": True},
                      debug_summary={"snapshot_marker": marker})
        )


def make_session():
    session = StubFinalAnalysisSession(model=None, device=None, db_operational={}, weights_cfg={})
    session.raw2d_buffer = [np.full((18, 2), 1.0)]
    session.visibility_buffer = [{}]
    session.aligned_seq = [np.full((18, 3), 1.0)]
    return session


class FinalAnalysisNonBlockingTests(unittest.TestCase):
    def tearDown(self):
        for session in getattr(self, "sessions", []):
            session.close_final_analysis()
            session.close_scoring()
            session.close_diagnostics()

    def track(self, session):
        self.sessions = getattr(self, "sessions", []) + [session]
        return session

    def test_submit_returns_before_final_analysis_finishes(self):
        session = self.track(make_session())
        started = time.perf_counter()
        session._finalize_rep(0)
        self.assertLess(time.perf_counter() - started, 0.03)

    def test_single_worker_preserves_rep_order(self):
        session = self.track(make_session())
        session._finalize_rep(0)
        session.raw2d_buffer[0][:] = 2.0
        session._finalize_rep(1)
        session.close_final_analysis()
        self.assertEqual([rep.rep_index for rep in session.completed_reps], [0, 1])
        self.assertEqual([rep.debug_summary["snapshot_marker"] for rep in session.completed_reps], [1.0, 2.0])

    def test_snapshot_isolated_from_later_live_buffer_mutation(self):
        session = self.track(make_session())
        session._finalize_rep(0)
        session.raw2d_buffer[0][:] = 99.0
        session.close_final_analysis()
        self.assertEqual(session.completed_reps[0].debug_summary["snapshot_marker"], 1.0)

    def test_shutdown_waits_for_pending_analysis_without_losing_result(self):
        session = self.track(make_session())
        session._finalize_rep(0)
        session.close_final_analysis()
        self.assertEqual(len(session.completed_reps), 1)
        self.assertTrue(session.final_analysis_timings)


if __name__ == "__main__":
    unittest.main()
