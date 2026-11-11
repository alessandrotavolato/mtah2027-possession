import tempfile
from pathlib import Path
import unittest
import numpy as np
from external_adapter import parse_features,parse_labels

class ExternalAdapter(unittest.TestCase):
    def test_coordinates_and_label_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);match=p/'match.xml';positions=p/'positions.xml'
            match.write_text('<Root><General HomeTeamId="h" GuestTeamId="a"/><Environment PitchX="105" PitchY="68"/></Root>')
            def xml(status,possession):
                groups=[]
                for team,person,offset in [('h','h1',-1),('a','a1',1),('BALL','ball',0)]:
                    frames=[]
                    for i in range(3):
                        frames.append(f'<Frame N="{10+i}" T="2022-01-01T00:00:00.{i*40:03d}+00:00" X="{i+offset}" Y="0" BallStatus="{status}" BallPossession="{possession}"/>')
                    groups.append(f'<FrameSet GameSection="firstHalf" TeamId="{team}" PersonId="{person}">'+''.join(frames)+'</FrameSet>')
                return '<Root>'+''.join(groups)+'</Root>'
            positions.write_text(xml('1','1'));a=parse_features(positions,match)
            np.testing.assert_array_equal(a['firstHalf']['distance'],np.ones((3,2)))
            self.assertEqual(a['firstHalf']['fps'],25)
            self.assertAlmostEqual(a['firstHalf']['features'][1,2],25)
            self.assertTrue(np.all(parse_labels(positions,a)['firstHalf']=='H'))
            positions.write_text(xml('0','2'));b=parse_features(positions,match)
            np.testing.assert_allclose(a['firstHalf']['features'],b['firstHalf']['features'],equal_nan=True)
            self.assertTrue(np.all(parse_labels(positions,b)['firstHalf']=='D'))

if __name__=='__main__':unittest.main()
