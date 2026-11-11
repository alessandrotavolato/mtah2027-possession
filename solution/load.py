"""Parse the raw ATD once into numpy arrays and cache them.

players: (n_frames, n_slots, 3) x[m], y[m], speed  (NaN where missing)
ball:    (n_frames, 3)          x[m], y[m], z/speed (NaN where missing)
team:    (n_slots,)             0 home, 1 away
Row i is video frame i (ATD frame_id - 1).
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAMES, GAME, require, source_frames  # noqa: E402

CACHE = os.path.join(ROOT, "data", "cache_tactical.npz")


def team_of_slots(meta_path):
    root = ET.parse(meta_path).getroot()
    home_id = [t.get("id") for t in root.iter("Team")][0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse(raw_path, n_frames, n_slots, pitch):
    L, W = pitch
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as fh:
        for line in fh:
            fid, pl, bl = line.rstrip("\n").split(":")
            i = int(fid) - 1
            vals = np.array(pl.replace(";", ",").split(","), dtype=np.float32)
            players[i] = vals.reshape(n_slots, 3)
            ball[i] = np.array(bl.split(","), dtype=np.float32)
    players[..., 0] *= L
    players[..., 1] *= W
    ball[:, 0] *= L
    ball[:, 1] *= W
    return players, ball


def load():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["players"], d["ball"], d["team"]
    meta, raw = require(["atd_meta", "atd_raw"])
    team = team_of_slots(meta)
    players, ball = parse(raw, source_frames(), len(team), GAMES[GAME]["pitch"])
    np.savez(CACHE, players=players, ball=ball, team=team)
    return players, ball, team


if __name__ == "__main__":
    p, b, t = load()
    print(p.shape, b.shape, t.shape, "home slots", int((t == 0).sum()))
