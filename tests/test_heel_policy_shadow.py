import math
import unittest

from ai_trainer.heel_policy_shadow import (
    HEEL, NORMAL, UNKNOWN, apply_policy, classification_metrics,
    empirical_evidence, evidence_table, relative_heel_margin,
)
from ai_trainer.heel_semantic import validate_heel_candidate


class HeelPolicyShadowTests(unittest.TestCase):
    def test_policy_a(self): self.assertEqual(apply_policy("A_RAW_DTW", HEEL, "NO")[0], HEEL)
    def test_policy_b_rejects_no(self): self.assertEqual(apply_policy("B_NO_REJECT", HEEL, "NO")[0], UNKNOWN)
    def test_policy_b_keeps_ambiguous(self): self.assertEqual(apply_policy("B_NO_REJECT", HEEL, "AMBIGUOUS")[0], HEEL)
    def test_policy_c_requires_yes(self): self.assertEqual(apply_policy("C_YES_REQUIRED", HEEL, "AMBIGUOUS")[0], UNKNOWN)
    def test_policy_c_preserves_yes(self): self.assertEqual(apply_policy("C_YES_REQUIRED", HEEL, "YES")[0], HEEL)
    def test_no_never_becomes_normal(self): self.assertNotEqual(apply_policy("B_NO_REJECT", HEEL, "NO")[0], NORMAL)
    def test_nonheel_is_unchanged(self): self.assertEqual(apply_policy("C_YES_REQUIRED", NORMAL, "NO")[0], NORMAL)
    def test_existing_unknown_is_unchanged(self): self.assertEqual(apply_policy("E_PRODUCTION", UNKNOWN, "NO")[0], UNKNOWN)
    def test_direct_gate(self): self.assertEqual(apply_policy("D_HEEL_DIRECT_2CLASS", HEEL, "YES", NORMAL)[0], UNKNOWN)
    def test_production_reproduction(self):
        for evidence in ("NO", "AMBIGUOUS", "YES"):
            shadow = apply_policy("E_PRODUCTION", HEEL, evidence)
            production = validate_heel_candidate(HEEL, HEEL, {"verdict": evidence}, 0.1)
            self.assertEqual(shadow, production)
    def test_relative_margin(self): self.assertAlmostEqual(relative_heel_margin(4.0, 3.0), .25)
    def test_margin_rejects_nan(self):
        with self.assertRaises(ValueError): relative_heel_margin(math.nan, 1)
    def test_unknown_metric(self):
        m=classification_metrics([{"truth":NORMAL,"result":UNKNOWN},{"truth":HEEL,"result":HEEL}])
        self.assertEqual(m["unknown_n"],1); self.assertEqual(m["unknown_rate"],.5)
    def test_normal_to_heel_count(self):
        m=classification_metrics([{"truth":NORMAL,"result":HEEL},{"truth":HEEL,"result":HEEL}])
        self.assertEqual(m["normal_to_heel_fp"],1)
    def test_heel_precision_recall(self):
        m=classification_metrics([{"truth":NORMAL,"result":HEEL},{"truth":HEEL,"result":HEEL}])
        self.assertEqual(m["heel_precision"],.5);self.assertEqual(m["heel_recall"],1)
    def test_evidence_tables_support_korean_labels(self):
        rows=[{"truth":NORMAL,"evidence":"NO","raw_prediction":HEEL}]
        self.assertEqual(evidence_table(rows)[0]["count"],1)
    def test_empirical_probability(self):
        rows=[{"truth":NORMAL,"evidence":"YES"},{"truth":HEEL,"evidence":"YES"}]
        self.assertEqual(empirical_evidence(rows)[2]["p_heel"],.5)


if __name__ == "__main__": unittest.main()
