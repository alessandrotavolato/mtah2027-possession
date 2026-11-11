"""Coordinate search on the EXTERNAL games only (never the competition match's labels)."""
import json, os, sys
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import validate_external as ve


def run(p):
    r = ve.score(p)
    return float(np.mean([x[0] for x in r])), [round(x[0], 3) for x in r], [x[2] for x in r]


if __name__ == "__main__":
    base = json.loads(sys.argv[1])
    grid = json.loads(sys.argv[2])            # {"param": [values]}
    jobs = [(k, v, dict(base, **{k: v})) for k, vals in grid.items() for v in vals]
    with Pool(min(8, len(jobs))) as pool:
        res = pool.map(run, [j[2] for j in jobs])
    for (k, v, _), (m, per, poss) in zip(jobs, res):
        print("{:>14} = {:<8} mean {:.4f}  per-game {}  poss {}".format(k, str(v), m, per, poss))
