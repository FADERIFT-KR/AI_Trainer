import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from types import SimpleNamespace
from dataclasses import replace
import numpy as np
from PyQt5.QtWidgets import QApplication
from ai_trainer.game_ui.pipeline_worker import PipelineStatus
from ai_trainer.game_ui.screens import CompareScreen
from ai_trainer.pose_sampling import PoseSampler
from ai_trainer.render import fit_transform


class PanelSourcesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_update_hold_update_and_disabled(self):
        screen = CompareScreen()
        coords = np.arange(54, dtype=float).reshape(18, 3)/54
        screen.ref_track = SimpleNamespace()
        screen._ref_tf = fit_transform(coords[None,:,:2],480,480)
        a = PipelineStatus(np.zeros((40,40,3),np.uint8),10,True,1,0,False,'',None,None,0,None,None,None,None,coords,
                           diagnostic_enabled=True,observation_id=10,sample_index=30,sample_timestamp=1.0)
        try:
            screen._on_status(a)
            self.assertIn('갱신',screen.skeleton_diagnostic.text())
            self.assertEqual(screen._displayed_skeleton_source,(10,30,1.0))
            image = screen.my_skeleton_panel._image.copy()
            b = replace(a,video_bgr=np.full((40,40,3),255,np.uint8),aligned_frame=None,observation_id=11,sample_index=33,sample_timestamp=1.1)
            screen._on_status(b)
            self.assertIn('observation_id=11',screen.camera_diagnostic.text())
            self.assertIn('유지',screen.skeleton_diagnostic.text())
            self.assertEqual(screen._displayed_skeleton_source,(10,30,1.0))
            self.assertEqual(image,screen.my_skeleton_panel._image)
            self.assertEqual(screen.camera_panel._image.pixelColor(0,0).red(),255)
            c = replace(a,aligned_frame=coords*.7,observation_id=12,sample_index=36,sample_timestamp=1.2)
            screen._on_status(c)
            self.assertEqual(screen._displayed_skeleton_source,(12,36,1.2))
            self.assertNotEqual(image,screen.my_skeleton_panel._image)
            self.assertIn('갱신',screen.skeleton_diagnostic.text())
            screen._on_status(replace(c,diagnostic_enabled=False))
            self.assertTrue(screen.camera_diagnostic.isHidden())
            self.assertTrue(screen.skeleton_diagnostic.isHidden())
        finally:
            screen.stop()
            screen.close()

    def test_sampler_coordinates_unchanged_and_real_clock(self):
        sampler=PoseSampler()
        a=np.zeros((18,3));b=np.ones((18,3))
        np.testing.assert_array_equal(sampler.push(a,10.0)[0],a)
        self.assertEqual(sampler.sample_timestamps,[10.0])
        samples=sampler.push(b,10.1)
        self.assertEqual(len(samples),3)
        np.testing.assert_allclose(sampler.sample_timestamps,[10+1/30,10+2/30,10.1])
        for sample,t in zip(samples,sampler.sample_timestamps):
            np.testing.assert_allclose(sample,(t-10)/.1*b)
        self.assertEqual(sampler.push(b,10.11),[])
        self.assertEqual(sampler.sample_timestamps,[])
        sampler.push(a,11.0)
        self.assertEqual(sampler.sample_timestamps,[11.0])


if __name__=='__main__':
    unittest.main()
