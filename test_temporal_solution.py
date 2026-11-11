import unittest
import numpy as np
from temporal_solution import decode

class TemporalBehavior(unittest.TestCase):
    def inputs(self,n=100):
        return np.tile([0.5,8.],(n,1)),np.zeros(n,bool),tuple(np.zeros(n,bool) for _ in range(3))
    def test_one_noisy_opposition_frame_does_not_change_team(self):
        d,o,m=self.inputs();d[50]=[8.,0.5]
        self.assertTrue(np.all(decode(d,o,m,False)=='H'))
    def test_sustained_opponent_control_changes_team(self):
        d,o,m=self.inputs();d[50:]=[8.,0.5]
        labels=decode(d,o,m,False)
        self.assertEqual(labels[20],'H');self.assertEqual(labels[80],'A')
    def test_absent_evidence_is_unknown(self):
        d,o,m=self.inputs();d[:]=np.nan
        self.assertTrue(np.all(decode(d,o,m,False)=='X'))
    def test_outside_heuristic_requires_persistence(self):
        d,o,m=self.inputs();o[10:12]=True;o[50:65]=True
        labels=decode(d,o,m,False)
        self.assertTrue(np.all(labels[10:12]=='H'))
        self.assertTrue(np.all(labels[50:65]=='D'))

if __name__=='__main__':
    unittest.main()
