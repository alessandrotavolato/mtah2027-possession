import tempfile
import unittest
from pathlib import Path
import numpy as np
from motion_features import dfl,velocity

class MotionFeatures(unittest.TestCase):
    def test_vector_difference_distinguishes_same_from_opposite_motion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'match.xml').write_text('<root><General HomeTeamId="H" GuestTeamId="A"/><Environment PitchX="105" PitchY="68"/></root>')
            xml=['<root>']
            for team,xs in [('H',[0,.2,.4]),('A',[5,4.8,4.6])]:
                xml.append('<FrameSet TeamId="'+team+'" PersonId="p'+team+'" GameSection="firstHalf">')
                for i,x in enumerate(xs):xml.append(f'<Frame N="{i}" X="{x}" Y="0"/>')
                xml.append('</FrameSet>')
            xml.append('</root>');(root/'positions.xml').write_text(''.join(xml))
            base={'features':np.zeros((3,6)),'fps':25.,'half':np.array(['firstHalf']*3),'frame_ids':np.arange(3),'ball_xy':np.array([[0,0],[.2,0],[.4,0]])}
            out=dfl(base,root/'positions.xml',root/'match.xml')
            self.assertTrue(np.isnan(out[0]).all())
            np.testing.assert_allclose(out[1:,0],0,atol=1e-10)
            np.testing.assert_allclose(out[1:,1],10,atol=1e-10)
            np.testing.assert_allclose(out[1:,2:],5,atol=1e-10)
    def test_jump_and_missing_observations_are_withheld(self):
        self.assertTrue(np.isnan(velocity(np.array([5.,0]),np.array([0.,0]),25,15)).all())
        self.assertTrue(np.isnan(velocity(np.array([np.nan,0]),np.array([0.,0]),25,15)).all())

if __name__=='__main__':unittest.main()
