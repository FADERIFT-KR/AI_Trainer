import unittest
import numpy as np
from ai_trainer.reference_quality_audit import audit_angle_trajectory,sensitivity

H,K=160,130
def audit(h,k):return audit_angle_trajectory(np.asarray(h,float),np.asarray(k,float),hip_standing_min=H,knee_standing_min=K)
def squat(n=41):
    x=np.sin(np.linspace(0,np.pi,n));return 175-100*x,175-120*x

class ReferenceQualityAuditTests(unittest.TestCase):
    def test_full_squat_passes(self):self.assertTrue(audit(*squat())["full_rep_candidate"])
    def test_bottom_start_is_incomplete(self):
        h,k=squat();r=audit(h[len(h)//2:],k[len(k)//2:]);self.assertNotEqual(r["quality"],"PASS")
    def test_no_return_is_incomplete(self):
        h,k=squat();r=audit(h[:25],k[:25]);self.assertFalse(r["returns_to_standing"])
    def test_boundary_bottom_flag(self):
        h=np.linspace(175,70,20);k=np.linspace(175,50,20);self.assertFalse(audit(h,k)["bottom_not_at_boundary"])
    def test_descend_only(self):
        r=audit(np.linspace(175,70,20),np.linspace(175,50,20));self.assertFalse(r["has_ascent"])
    def test_ascend_only(self):
        r=audit(np.linspace(70,175,20),np.linspace(50,175,20));self.assertFalse(r["has_descent"])
    def test_flat(self):self.assertFalse(audit(np.ones(20)*175,np.ones(20)*175)["has_bottom"])
    def test_two_reps(self):
        x=np.sin(np.linspace(0,2*np.pi,81))**2;r=audit(175-100*x,175-120*x);self.assertIn("MULTIPLE_REP_CANDIDATE",r["reasons"])
    def test_nan(self):
        with self.assertRaises(ValueError):audit([170]*7+[np.nan],[170]*8)
    def test_inf(self):
        with self.assertRaises(ValueError):audit([170]*7+[np.inf],[170]*8)
    def test_short(self):
        with self.assertRaises(ValueError):audit([170]*7,[170]*7)
    def test_quality_has_no_prediction_argument(self):self.assertNotIn("prediction",audit(*squat()))
    def test_pass_sensitivity_counts(self):
        rows=[{"quality":"PASS","actual_class":"정상","methods":{"DTW":{"predicted_class":"정상"}}},{"quality":"SEVERE","actual_class":"오류","methods":{"DTW":{"predicted_class":"정상"}}}]
        r=sensitivity(rows,"DTW");self.assertEqual((r["evaluated"],r["excluded"]),(1,1))
    def test_empty_pass_subset_safe(self):
        r=sensitivity([{"quality":"SEVERE","actual_class":"정상","methods":{}}],"DTW");self.assertIsNone(r["accuracy"])
    def test_korean_identifier_is_irrelevant_to_geometry(self):self.assertTrue(audit(*squat())["full_rep_candidate"])

if __name__=="__main__":unittest.main()
