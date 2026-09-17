import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ai_trainer.common_skeleton import COMMON_JOINT_NAMES
from ai_trainer.heel_shadow_validation import HeelShadowValidationLogger, direct_heel_features

I = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def sequence():
    p=np.zeros((8,18,2),float);p[:,I["Neck"],1]=0;p[:,I["Hip"],1]=100
    p[:,I["LHeel"],1]=200;p[:,I["RHeel"],1]=200
    p[:,I["LBigToe"],1]=210;p[:,I["RBigToe"],1]=210
    p[5:,I["LHeel"],1]+=[5,10,20];p[5:,I["RHeel"],1]+=[10,20,30]
    return p


class FeatureTests(unittest.TestCase):
    def test_heel_toe_range_and_aggregation(self):
        f=direct_heel_features(sequence())
        self.assertAlmostEqual(f["left_heel_toe_range"],.2)
        self.assertAlmostEqual(f["right_heel_toe_range"],.3)
        self.assertAlmostEqual(f["heel_toe_range_mean"],.25)

    def test_vertical_displacement_and_normalization(self):
        f=direct_heel_features(sequence())
        self.assertAlmostEqual(f["torso_length_px"],100)
        self.assertAlmostEqual(f["left_heel_vertical_max_abs"],.2)
        self.assertAlmostEqual(f["right_heel_vertical_max_abs"],.3)
        self.assertAlmostEqual(f["heel_vertical_max_abs_mean"],.25)


class LoggerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.logger=HeelShadowValidationLogger(self.root,now=datetime(2026,1,2,3,4,5))
        self.production={"raw_3d_dtw":"발뒤꿈치오류","final":"발뒤꿈치오류","reason_code":"VALIDATED","class_distances":{"정상":2,"발뒤꿈치오류":1,"엉덩이하방오류":3,"고관절오류":4}}
        self.thresholds={"no_evidence_max":.08,"positive_evidence_min":.1}
    def tearDown(self): self.logger.close();self.tmp.cleanup()
    def result(self,index=0): return SimpleNamespace(rep_index=index,debug_summary={"heel_semantic":{"value":.09,"verdict":"AMBIGUOUS"}})
    def record(self,index=0): return self.logger.safe_record(result=self.result(index),production=self.production,raw2d=sequence(),heel_thresholds=self.thresholds)

    def test_gt_is_blank_and_prediction_is_separate(self):
        row=self.record();self.assertEqual(row["ground_truth_label"],"");self.assertEqual(row["ground_truth_note"],"");self.assertNotEqual(row["ground_truth_label"],row["final_production_class"])
    def test_rep_ordering(self):
        self.record(0);self.record(1);self.assertEqual([r["rep_id"] for r in self.logger.rows],[1,2])
    def test_logger_failure_is_isolated(self):
        before=dict(self.production)
        with patch.object(self.logger,"_write_all",side_effect=OSError("disk")):
            self.assertIsNone(self.record())
        self.assertEqual(self.production,before);self.assertTrue(self.logger.failures)
    def test_shadow_does_not_change_final_classification(self):
        before=dict(self.production);row=self.record();self.assertEqual(self.production,before);self.assertEqual(row["final_production_class"],before["final"])
    def test_csv_and_summary_exist(self):
        self.record();self.assertTrue((self.logger.path/"reps.csv").exists());self.assertTrue((self.logger.path/"session_summary.json").exists())
        with (self.logger.path/"annotation_template.csv").open(encoding="utf-8-sig") as stream:
            row=next(csv.DictReader(stream));self.assertEqual(row["ground_truth_label"],"")


if __name__=="__main__": unittest.main()
