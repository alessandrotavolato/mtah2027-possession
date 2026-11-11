"""Regenerate the submission.

    python carl/run.py            -> evaluation/results/carl_possession.csv

Needs the BAR-ATH tactical bundle in data/ (or MTAH_DATA_DIR) and Metrica's public sample data
(github.com/metrica-sports/sample-data) in EXTERNAL_DIR (default ~/code/external/sample-data),
which trains the dead-ball classifier. Fully deterministic (fixed seeds).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from starter.baseline import write_submission  # noqa: E402
import model  # noqa: E402
from match import barca  # noqa: E402
from features import compute  # noqa: E402

EXTERNAL_GAMES = (1, 2)


def stats(labels, blocks):
    spells = []
    for _, a, b in blocks:
        w = labels[a:b + 1]
        br = np.flatnonzero(w[1:] != w[:-1]) + 1
        bounds = np.r_[0, br, len(w)]
        for i in range(len(bounds) - 1):
            spells.append((w[bounds[i]], bounds[i + 1] - bounds[i]))
    poss = [l for k, l in spells if k in "HA"]
    dead = [l for k, l in spells if k == "D"]
    print("possessions {}  mean {:.1f}s  | dead spells {} mean {:.1f}s | D share {:.1%}".format(
        len(poss), np.mean(poss) / 25, len(dead), np.mean(dead) / 25 if dead else 0,
        np.mean(labels[labels != "X"] == "D")))


def train_dead_clf(p):
    import dead_model
    from external import load_game
    rows = []
    for k in EXTERNAL_GAMES:
        M, gt = load_game(k)
        rows.append(dead_model.training_rows(M, compute(M, p["interp_gap_s"]), gt))
    return dead_model.train(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="carl_possession")
    args = ap.parse_args()
    p = dict(model.PARAMS)
    clf = train_dead_clf(p)
    M = barca()
    labels = model.predict(M, p, dead_clf=clf)
    stats(labels, M["blocks"])
    path = write_submission(os.path.join(ROOT, "evaluation", "results", args.name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT))
