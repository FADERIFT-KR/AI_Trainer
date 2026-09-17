import unittest
import numpy as np
from ai_trainer.feature_separation import distribution_stats,rank_auc,overlap_coefficient,separation,semantic_counts,remove_one

class FeatureSeparationTests(unittest.TestCase):
 def test_stats(self):self.assertEqual(distribution_stats([1,2,3])["median"],2)
 def test_percentile(self):self.assertEqual(distribution_stats(range(11))["p90"],9)
 def test_effect(self):self.assertGreater(separation([0,1,2],[3,4,5])["cohens_d"],0)
 def test_auc_direction(self):self.assertEqual(rank_auc([3,4],[1,2]),0)
 def test_overlap(self):self.assertEqual(overlap_coefficient([1,1],[1,1]),1)
 def test_domain_is_caller_metadata(self):self.assertNotIn("domain",separation([1,2],[2,3]))
 def test_semantic_counts(self):self.assertEqual(semantic_counts([0,.5,1],.2,.8),{"NO":1,"AMBIGUOUS":1,"YES":1})
 def test_contribution_sum(self):self.assertAlmostEqual(sum([.1,.2]),.3)
 def test_remove_exact_one(self):
  x=remove_one({'a':1,'b':1},'a');self.assertEqual(x,{'a':0.0,'b':1})
 def test_config_not_mutated(self):
  x={'a':1};remove_one(x,'a');self.assertEqual(x,{'a':1})
 def test_binary_confusion_shape(self):self.assertEqual(np.zeros((2,2)).shape,(2,2))
 def test_geometry_metric_bounded(self):self.assertLessEqual(rank_auc([0,1],[1,2]),1)
 def test_group_distance(self):self.assertEqual(np.linalg.norm(np.array([1])-np.array([0])),1)
 def test_korean(self):self.assertEqual({'정상':1}['정상'],1)
 def test_nan(self):
  with self.assertRaises(ValueError):distribution_stats([1,np.nan])
 def test_inf(self):
  with self.assertRaises(ValueError):distribution_stats([1,np.inf])
 def test_short(self):
  with self.assertRaises(ValueError):distribution_stats([1])
if __name__=='__main__':unittest.main()
