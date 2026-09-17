import csv
import json
import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from ai_trainer.live_heel_shadow import LiveHeelShadowLogger, SCHEMA, _two_class_gate
from scripts.evaluate_live_heel_shadow import evaluate


class LiveHeelShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.logger = LiveHeelShadowLogger(self.root, now=datetime(2026,1,2,3,4,5))
        self.production={"raw_3d_dtw":"발뒤꿈치오류","pre_heel_final":"발뒤꿈치오류","final":"발뒤꿈치오류","reason_code":"VALIDATED",
                         "class_distances":{"정상":4.0,"발뒤꿈치오류":3.0},"depth_2d_pass":True,"consistency_pass":True}
        self.two_d={"distance_by_class":{"정상":1.0,"발뒤꿈치오류":2.0},"semantic":{"heel_relative_motion":.1}}
        self.result=SimpleNamespace(rep_index=0,debug_summary={"heel_semantic":{"verdict":"NO","value":.07,"left":{},"right":{}}})
        self.thresholds={"no_evidence_max":.08,"positive_evidence_min":.1}
    def tearDown(self): self.logger.close();self.tmp.cleanup()
    def record(self, **kw): return self.logger.record(result=self.result,production=self.production,two_d=self.two_d,heel_thresholds=self.thresholds,**kw)
    def test_shadow_b(self): self.assertEqual(self.record()["shadow_b_final"],"자세추정불확실")
    def test_shadow_d(self): self.assertEqual(self.record()["shadow_d_final"],"자세추정불확실")
    def test_no_conflict(self): self.assertIn("CONFLICT",self.record()["shadow_b_reason"])
    def test_ambiguous(self): self.result.debug_summary["heel_semantic"]["verdict"]="AMBIGUOUS";self.assertEqual(self.record()["shadow_b_final"],"발뒤꿈치오류")
    def test_yes(self): self.result.debug_summary["heel_semantic"]["verdict"]="YES";self.assertEqual(self.record()["shadow_b_final"],"발뒤꿈치오류")
    def test_nonheel_passthrough(self):
        self.production.update(raw_3d_dtw="고관절오류",pre_heel_final="자세추정불확실",final="자세추정불확실")
        self.assertEqual(self.record()["shadow_d_final"],"자세추정불확실")
    def test_prior_semantic_unknown_is_preserved_for_raw_heel(self):
        self.production.update(pre_heel_final="자세추정불확실",final="자세추정불확실",reason_code="UNKNOWN_DEPTH_CONFLICT")
        self.assertEqual(self.record()["shadow_d_final"],"자세추정불확실")
    def test_production_untouched(self): before=dict(self.production);self.record();self.assertEqual(self.production,before)
    def test_ground_truth_default_blank(self): self.assertEqual(self.record()["ground_truth_label"],"")
    def test_prediction_not_copied_to_truth(self): self.assertNotEqual(self.record()["ground_truth_label"],self.production["final"])
    def test_csv_schema(self):
        self.record()
        with (self.logger.path/'reps.csv').open(encoding='utf-8-sig') as stream:
            self.assertEqual(next(csv.reader(stream)),SCHEMA)
    def test_jsonl_schema(self): self.record();row=json.loads((self.logger.path/'rep_details.jsonl').read_text(encoding='utf-8').splitlines()[0]);self.assertTrue(set(SCHEMA)<=set(row))
    def test_disagreement(self): self.assertTrue(self.record()["production_vs_d_disagree"])
    def test_summary(self): self.record();self.assertEqual(self.logger.summary()["raw_heel_candidates"],1)
    def test_missing_distance(self): self.two_d={"distance_by_class":{}};self.assertIsNone(self.record()["heel_2d_prediction"])
    def test_nan(self): self.two_d={"distance_by_class":{"정상":math.nan,"발뒤꿈치오류":1}};self.assertIsNone(_two_class_gate(self.two_d)[2])
    def test_korean(self): self.assertEqual(self.record()["raw_class"],"발뒤꿈치오류")
    def test_async_integration(self):
        with ThreadPoolExecutor(max_workers=1) as ex: row=ex.submit(self.record,two_d_compute_ms=2).result()
        self.assertGreaterEqual(row["shadow_compute_ms"],2)
    def test_manual_eval_excludes_blank(self):
        self.record();result=evaluate(self.logger.path);self.assertEqual(result["evaluated_binary_n"],0)


if __name__=="__main__":unittest.main()
