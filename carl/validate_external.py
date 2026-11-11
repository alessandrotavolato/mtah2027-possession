"""Score model variants on the external sample games with the official metric code.

    python carl/validate_external.py '{"name": {param overrides}, ...}'
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, HERE)
import possession_metrics as pm  # noqa: E402
from external import load_game  # noqa: E402
from features import compute  # noqa: E402
import model  # noqa: E402

GAMES = (1, 2)
_cache = {}
NOISE = bool(int(os.environ.get("EXT_NOISE", "0")))


def degrade(M, seed=0, drop_frac=0.07, seg_s=8.0, ball_sigma_m=0.05):
    """Make clean sample tracking look like automatic ATD: player tracks drop out in segments,
    ball jitters. Fixed seed, so validation is reproducible."""
    rng = np.random.default_rng(seed)
    P = M["P"].copy()
    B = M["B"].copy()
    n, S, _ = P.shape
    seg = int(seg_s * 25)
    n_seg = int(drop_frac * n * S / seg)
    for _ in range(n_seg):
        j = rng.integers(S)
        a = rng.integers(n)
        P[a:a + rng.integers(seg // 2, 2 * seg), j, :] = np.nan
    B[:, 0] += rng.normal(0, ball_sigma_m / M["length"], n).astype(np.float32)
    B[:, 1] += rng.normal(0, ball_sigma_m / M["width"], n).astype(np.float32)
    return dict(M, P=P, B=B)


def game(k, interp):
    key = (k, interp)
    if key not in _cache:
        if k not in _cache:
            M, gt = load_game(k)
            if NOISE:
                M = degrade(M, seed=k)
            _cache[k] = (M, gt)
        M, gt = _cache[k]
        _cache[key] = (M, gt, compute(M, interp))
    return _cache[key]


_clfs = {}


def dead_clf_excluding(k, interp, novis=False):
    """Dead-ball classifier trained on every external game except k (leave-one-game-out)."""
    key = (k, interp, novis)
    if key not in _clfs:
        import dead_model
        drop = dead_model.VISIBILITY if novis else ()
        rows = [dead_model.training_rows(game(j, interp)[0], game(j, interp)[2],
                                         game(j, interp)[1], drop=drop) for j in GAMES if j != k]
        _clfs[key] = dead_model.train(rows)
    return _clfs[key]


def score(p, games=GAMES):
    out = []
    p = dict(p)
    use_dead = p.pop("dead_clf", False)
    for k in games:
        interp = dict(model.PARAMS, **p)["interp_gap_s"]
        M, gt, F = game(k, interp)
        novis = dict(model.PARAMS, **p)["dead_no_visibility"]
        clf = dead_clf_excluding(k, interp, novis) if use_dead else None
        lab = model.predict(M, p, F, dead_clf=clf)
        g = pm.label_codes(gt)
        r = pm.evaluate(g, pm.label_codes(lab))
        runs = pm.valid_runs(g)
        out.append((r["spell"]["mean_f1"] if "spell" in r else r["spell_f1_mean"], r["macro_f1"],
                    len(pm.possession_spells(pm.label_codes(lab), runs)),
                    len(pm.possession_spells(g, runs))))
    return out


if __name__ == "__main__":
    variants = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"default": {}}
    for name, p in variants.items():
        res = score(p)
        s = " | ".join("g{} spell {:.3f} macro {:.3f} poss {}/{}".format(k, *r) for k, r in zip(GAMES, res))
        print("{:<18} mean spell {:.3f}  {}".format(name, np.mean([r[0] for r in res]), s), flush=True)
