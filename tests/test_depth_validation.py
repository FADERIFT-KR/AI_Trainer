import unittest

from ai_trainer.online_dtw import OnlineSquatSession


class DepthValidationTests(unittest.TestCase):
    def setUp(self):
        self.session = OnlineSquatSession(
            model=None,
            device=None,
            db_operational={},
            weights_cfg={},
        )
        self.session.normal_depth_threshold = 0.425
        self.distances = {"정상": 0.20, "엉덩이하방오류": 0.10, "고관절오류": 0.30}

    def test_shallow_class_waits_until_bottom_or_ascent(self):
        self.session.rep_min_pelvis_height = 0.70

        self.assertIsNone(self.session._depth_aware_class(self.distances, depth_ready=False))

    def test_deep_enough_rep_rejects_shallow_class(self):
        self.session.rep_min_pelvis_height = 0.40

        self.assertEqual(
            self.session._depth_aware_class(self.distances, depth_ready=True),
            "정상",
        )

    def test_genuinely_shallow_rep_keeps_shallow_class(self):
        self.session.rep_min_pelvis_height = 0.70

        self.assertEqual(
            self.session._depth_aware_class(self.distances, depth_ready=True),
            "엉덩이하방오류",
        )


if __name__ == "__main__":
    unittest.main()
