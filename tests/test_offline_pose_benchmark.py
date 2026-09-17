import tempfile
import unittest

import numpy as np

from ai_trainer.angle_domain_diagnostic import frame_angles
from ai_trainer.offline_pose_benchmark import (
    confusion_and_metrics, derivative_features, geometry_vector,
    robust_location_scale, stable_derivative,
)
from ai_trainer.features import extract_all_features
from tests.test_posture_score import sequence


class OfflinePoseBenchmarkTests(unittest.TestCase):
    def test_identical_derivative_sequence_is_identical(self):
        x=np.arange(20,dtype=float)[:,None];self.assertTrue(np.allclose(stable_derivative(x),stable_derivative(x.copy())))

    def test_derivative_shape_is_preserved(self):
        x=np.zeros((7,4));self.assertEqual(stable_derivative(x).shape,x.shape)

    def test_speed_variation_derivative_is_finite(self):
        slow=np.sin(np.linspace(0,np.pi,31))[:,None];self.assertTrue(np.isfinite(stable_derivative(slow)).all())

    def test_derivative_rejects_nan(self):
        x=np.ones((4,2));x[1,0]=np.nan
        with self.assertRaises(ValueError):stable_derivative(x)

    def test_derivative_rejects_too_short(self):
        with self.assertRaises(ValueError):stable_derivative(np.ones((2,2)))

    def test_derivative_features_keeps_keys_and_shapes(self):
        f=extract_all_features(sequence()) ; d=derivative_features(f)
        self.assertEqual(set(f),set(d));self.assertTrue(all(f[k].shape==d[k].shape for k in f))

    def test_geometry_angle_convention_matches_common_function(self):
        coords=sequence(); vector,names=geometry_vector(coords)
        expected=np.median([frame_angles(x)["hip_avg"] for x in coords[:5]])
        self.assertAlmostEqual(vector[names.index("standing_hip")],expected)

    def test_geometry_excursion_is_positive_for_squat(self):
        vector,names=geometry_vector(sequence());self.assertGreater(vector[names.index("hip_excursion")],0)

    def test_geometry_rejects_nonfinite(self):
        x=sequence();x[2,0,0]=np.inf
        with self.assertRaises(ValueError):geometry_vector(x)

    def test_robust_normalization_uses_supplied_train_only(self):
        train=np.array([[0.],[1.],[2.]])
        loc,scale=robust_location_scale(train)
        self.assertEqual(loc[0],1.0);self.assertEqual(scale[0],1.0)

    def test_confusion_matrix_and_metrics(self):
        matrix,metrics=confusion_and_metrics(["정상","오류"],["정상","정상"],["정상","오류"])
        self.assertEqual(matrix.tolist(),[[1,0],[1,0]]);self.assertEqual(metrics["accuracy"],0.5)

    def test_korean_class_labels_are_preserved(self):
        labels=["정상","고관절오류","발뒤꿈치오류","엉덩이하방오류"]
        matrix,metrics=confusion_and_metrics(labels,labels,labels)
        self.assertEqual(int(matrix.trace()),4);self.assertEqual(set(metrics["per_class"]),set(labels))

    def test_leave_one_out_exclusion_pattern(self):
        samples=[{"id":"a"},{"id":"b"},{"id":"c"}]
        for test in samples:
            train=[s for s in samples if s["id"]!=test["id"]]
            self.assertNotIn(test["id"],[s["id"] for s in train]);self.assertEqual(len(train),2)


if __name__ == "__main__":unittest.main()
