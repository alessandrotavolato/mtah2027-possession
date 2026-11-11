import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import target_adapter


class RecoveryIntegration(unittest.TestCase):
    def test_recovered_position_changes_distances_and_records_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "meta.xml").write_text('<root><Team id="h"/><Team id="a"/><Player teamId="h"/><Player teamId="a"/></root>')
            (root / "raw.txt").write_text("\n".join(
                f"{i+1}:0.2,0.2,0;0.8,0.8,0:" + ("NaN,NaN,NaN" if i == 2 else "0.2,0.2,0")
                for i in range(5)))
            recovery = root / "recovery.csv"
            recovery.write_text("frame,x_m,y_m,confidence\n2,8,8,0.8\n")
            config = {"pitch": (10,10), "fps": 25, "n_video_frames": 5}
            with patch.object(target_adapter, "require", return_value=[root/"meta.xml",root/"raw.txt"]), patch.dict(target_adapter.GAMES, {target_adapter.GAME: config}):
                values = target_adapter.target_features(str(recovery))
            self.assertEqual(values["ball_source"][2], "video")
            self.assertEqual(values["video_confidence"][2], 0.8)
            self.assertEqual(values["distance"][2,1], 0)
            self.assertEqual(values["ball_source"][0], "tracking")
            self.assertTrue(np.isnan(values["video_confidence"][0]))

    def test_duplicate_recovery_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"r.csv"
            path.write_text("frame,x_m,y_m,confidence\n1,2,2,0.5\n1,2,2,0.5\n")
            with self.assertRaises(ValueError):
                target_adapter.read_recoveries(path,5)


if __name__ == "__main__":
    unittest.main()
