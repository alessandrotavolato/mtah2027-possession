import unittest
import numpy as np
from two_stage_models import probabilities,decode,occlude

class ConstantModel:
    def __init__(self,p):self.p=p
    def predict_proba(self,x):return np.tile(self.p,(len(x),1))

class TwoStageBehavior(unittest.TestCase):
    def test_joint_probabilities_have_correct_order_and_sum(self):
        b={'live':ConstantModel([.2,.8]),'team':ConstantModel([.75,.25])}
        np.testing.assert_allclose(probabilities(b,np.zeros((2,1))),[[.6,.2,.2],[.6,.2,.2]])
    def test_uniform_decoder_matches_long_sustained_evidence(self):
        p=np.tile([.95,.025,.025],(300,1));p[150:]=[.025,.95,.025]
        labels=decode(p,20,5)
        self.assertEqual(labels[25],0);self.assertEqual(labels[275],1)
    def test_occlusion_preserves_player_features(self):
        x=np.ones((1000,10));h=np.full(1000,'firstHalf')
        z=occlude(x,h,911)
        np.testing.assert_array_equal(z[:,4:6],x[:,4:6]);self.assertTrue(np.isnan(z[:,:4]).any())
        np.testing.assert_array_equal(np.isnan(z[:,6:]),np.repeat(np.isnan(z[:,0,None]),4,axis=1))

if __name__=='__main__':unittest.main()
