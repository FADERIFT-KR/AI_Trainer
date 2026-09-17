import numpy as np
import unittest

from ai_trainer.adapter_diagnostic import adjust_hip_width_copy, geometry_ratios
from ai_trainer.common_skeleton import COMMON_JOINT_NAMES


I = {name: i for i, name in enumerate(COMMON_JOINT_NAMES)}


def _pose():
    p = np.zeros((18, 2), dtype=np.float64)
    p[I["Hip"]] = [10, 10]
    p[I["Neck"]] = [10, 0]
    p[I["LShoulder"]], p[I["RShoulder"]] = [7, 0], [13, 0]
    p[I["LHip"]], p[I["RHip"]] = [8, 10], [12, 10]
    p[I["LKnee"]], p[I["RKnee"]] = [8, 15], [12, 15]
    p[I["LAnkle"]], p[I["RAnkle"]] = [8, 20], [12, 20]
    p[I["LHeel"]], p[I["RHeel"]] = [7, 20], [13, 20]
    p[I["LBigToe"]], p[I["RBigToe"]] = [7, 21], [13, 21]
    return p


class HipGeometryDiagnosticTests(unittest.TestCase):
    def test_geometry_ratios_ignore_uniform_fit_and_translation(self):
        p = _pose()
        a = geometry_ratios(p)
        b = geometry_ratios(p * 3.7 + np.array([91.0, -12.0]))
        self.assertTrue(all(np.isclose(a[k], b[k]) for k in a))

    def test_hip_adjustment_preserves_input_and_each_connected_leg_chain(self):
        raw = np.stack([_pose(), _pose() + [0.5, 1.0]])
        before = raw.copy()
        adjusted = adjust_hip_width_copy(raw, 0.2)
        self.assertTrue(np.array_equal(raw, before))
        self.assertTrue(np.isclose(geometry_ratios(adjusted[0])["hip_width/torso_length"], 0.2))
        for side in ("L", "R"):
            for child, parent in (("Knee", "Hip"), ("Ankle", "Knee"), ("Heel", "Ankle"), ("BigToe", "Ankle")):
                old = raw[0, I[f"{side}{child}"]] - raw[0, I[f"{side}{parent}"]]
                new = adjusted[0, I[f"{side}{child}"]] - adjusted[0, I[f"{side}{parent}"]]
                self.assertTrue(np.allclose(old, new))
