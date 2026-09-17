import unittest

from ai_trainer.rep_phase_display import rep_phase_label


class RepPhaseDisplayTests(unittest.TestCase):
    def test_production_state_mapping(self):
        self.assertEqual(rep_phase_label("prep"), "준비")
        self.assertEqual(rep_phase_label("descend"), "하강")
        self.assertEqual(rep_phase_label("bottom"), "최저")
        self.assertEqual(rep_phase_label("ascend"), "상승")

    def test_rep_end_returns_to_prep_display(self):
        self.assertEqual(rep_phase_label("prep"), "준비")

    def test_unknown_and_initializing_states_are_safe(self):
        self.assertEqual(rep_phase_label(None, active=False), "준비 중")
        self.assertEqual(rep_phase_label(None, active=True), "자세 확인 중")

    def test_two_rep_display_sequence_is_natural(self):
        internal = ["prep", "descend", "bottom", "ascend", "prep"] * 2
        expected = ["준비", "하강", "최저", "상승", "준비"] * 2
        self.assertEqual([rep_phase_label(state) for state in internal], expected)


if __name__ == "__main__":
    unittest.main()
