import unittest
import numpy as np
from hybrid_solution import predict

class HybridBehavior(unittest.TestCase):
    def inputs(self, n=300):
        return np.tile([0.5, 8.], (n,1)), np.zeros(n,bool), tuple(np.zeros(n,bool) for _ in range(3))
    def test_isolated_wrong_observation_does_not_switch(self):
        d,o,m=self.inputs(); d[150]=[8.,0.5]
        self.assertTrue(np.all(predict(d,o,m)=='H'))
    def test_sustained_control_switches(self):
        d,o,m=self.inputs(); d[150:]=[8.,0.5]
        p=predict(d,o,m)
        self.assertEqual(p[50],'H');self.assertEqual(p[250],'A')
    def test_long_missing_gap_is_unknown_not_dead(self):
        d,o,m=self.inputs(600);d[100:500]=np.nan
        p=predict(d,o,m)
        self.assertEqual(p[300],'X');self.assertNotIn('D',p)
    def test_short_gap_interpolates_when_allowed(self):
        d,o,m=self.inputs();d[100:160]=np.nan
        self.assertEqual(predict(d,o,m)[130],'H')
    def test_restart_context_prevents_interpolation(self):
        d,o,m=self.inputs();d[100:160]=np.nan;m[2][100:160]=True
        self.assertEqual(predict(d,o,m)[130],'X')
    def test_outside_requires_sustained_observations(self):
        d,o,m=self.inputs();o[60:62]=True;o[150:170]=True
        p=predict(d,o,m)
        self.assertEqual(p[60],'H');self.assertTrue(np.all(p[150:170]=='D'))

if __name__=='__main__':unittest.main()
