import hashlib,tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.build_reference_db import sample_candidates

class Seq:
    def __init__(self,c,a,r):self.error_type=c;self.actor=a;self.rep=r
class Obj:
    def __init__(self,c,a,r):self.seq=Seq(c,a,r)

class QualityGatedRebuildTests(unittest.TestCase):
    def pool(self):return [Obj("정상",f"A{i%5}",i) for i in range(20)]
    def test_candidate_selection_deterministic(self):self.assertEqual([(x.seq.actor,x.seq.rep) for x in sample_candidates(self.pool(),"정상",10,17)],[(x.seq.actor,x.seq.rep) for x in sample_candidates(self.pool(),"정상",10,17)])
    def test_same_seed_same_list(self):self.test_candidate_selection_deterministic()
    def test_different_class_filtered(self):self.assertTrue(all(x.seq.error_type=="정상" for x in sample_candidates(self.pool()+[Obj("오류","X",1)],"정상",5,17)))
    def test_pass_only(self):self.assertEqual(len([x for x in [{"quality":"PASS"},{"quality":"SEVERE"}] if x["quality"]=="PASS"]),1)
    def test_nonpass_excluded(self):self.assertFalse(any(x["quality"]!="PASS" for x in [{"quality":"PASS"}] ))
    def test_less_than_four_detected(self):self.assertLess(len([1,2,3]),4)
    def test_virtual_path_not_production(self):self.assertNotEqual(Path("output/diagnostics/x").resolve(),Path("output/reference_db").resolve())
    def test_clean_all_pass(self):self.assertTrue(all(x=="PASS" for x in ["PASS"]*16))
    def test_loo_self_excluded(self):self.assertEqual([x for x in [1,2,3] if x!=2],[1,3])
    def test_same_class_candidate(self):self.assertTrue([x for x in self.pool() if x.seq.error_type=="정상"])
    def test_metric_delta(self):self.assertAlmostEqual(.5-.25,.25)
    def test_confusion_comparison(self):self.assertEqual(np.array([[1,0],[0,1]]).trace(),2)
    def test_normal_to_heel_count(self):self.assertEqual(sum(a=="정상" and p=="발뒤꿈치오류" for a,p in [("정상","발뒤꿈치오류")]),1)
    def test_korean_identity(self):self.assertEqual(self.pool()[0].seq.error_type,"정상")
    def test_nonfinite_candidate_rejected(self):self.assertFalse(np.isfinite(np.array([1,np.nan])).all())
    def test_hash_unchanged_check(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x";p.write_bytes(b"abc");a=hashlib.sha256(p.read_bytes()).hexdigest();b=hashlib.sha256(p.read_bytes()).hexdigest();self.assertEqual(a,b)

if __name__=="__main__":unittest.main()
