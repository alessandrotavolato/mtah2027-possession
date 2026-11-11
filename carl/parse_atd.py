"""Parse the raw ATD into dense numpy arrays, cached in data/ (gitignored)."""
import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import require, source_frames
from starter.baseline import team_of_each_track

CACHE = os.path.join(ROOT, "data", "cache_atd.npz")


def parse():
    meta, raw = require(["atd_meta", "atd_raw"])
    n = source_frames("tactical")
    team = team_of_each_track(meta)
    S = len(team)
    P = np.full((n, S, 3), np.nan, np.float32)
    B = np.full((n, 3), np.nan, np.float32)
    with open(raw) as fh:
        for line in fh:
            a, p, b = line.rstrip("\n").split(":")
            f = int(a) - 1
            for s, g in enumerate(p.split(";")):
                if g[0] != "N":
                    P[f, s] = [float(v) for v in g.split(",")]
            bb = b.split(";")[0]
            if bb[0] != "N":
                vals = [float(v) for v in bb.split(",")]
                B[f, :len(vals)] = vals[:3]
    np.savez_compressed(CACHE, P=P, B=B, team=team)


def load():
    if not os.path.exists(CACHE):
        parse()
    d = np.load(CACHE)
    return d["P"], d["B"], d["team"]


if __name__ == "__main__":
    P, B, team = load()
    print(P.shape, B.shape, team)
