"""Supervised dead-ball classifier, trained ONLY on external games (Metrica sample data).

Features are team-level and robust to tracking noise (no player identities):
  tempo at several scales and its change, max player speed, ball-seen fractions, time since /
  until the ball was last / next seen, where the ball was last / next seen relative to the touch
  and goal lines, ball outside the pitch, ball speed, nearest-player distance, player spread.
"""
import numpy as np

from features import rolling_mean

FPS = 25


def _since_until(seen):
    n = len(seen)
    since = np.full(n, 1e9)
    last = -1e9
    for i in range(n):
        if seen[i]:
            last = i
        since[i] = i - last
    until = np.full(n, 1e9)
    nxt = 1e9
    for i in range(n - 1, -1, -1):
        if seen[i]:
            nxt = i
        until[i] = nxt - i
    return np.minimum(since, 30 * FPS) / FPS, np.minimum(until, 30 * FPS) / FPS


def _ffill(v, seen):
    idx = np.where(seen, np.arange(len(v)), -1)
    idx = np.maximum.accumulate(idx)
    out = np.where(idx >= 0, v[np.maximum(idx, 0)], np.nan)
    return out


def _bfill(v, seen):
    return _ffill(v[::-1], seen[::-1])[::-1]


def line_distance(x, y, L, W):
    """(distance to nearest touchline, distance to nearest goal line), metres."""
    return np.minimum(y, W - y), np.minimum(x, L - x)


VISIBILITY = ("seen_1", "seen_3", "seen_8", "since", "until")


def features(M, F, drop=()):
    L, W = M["length"], M["width"]
    P = M["P"]
    seen = F["seen"]
    sp = F["mean_speed"]
    feats = {}
    for s in (0.5, 1, 3, 6):
        feats["tempo_%g" % s] = rolling_mean(sp, int(s * FPS))
    t1 = feats["tempo_1"]
    for lag in (2, 4):
        d = np.full(len(t1), np.nan)
        d[lag * FPS:] = t1[lag * FPS:] - t1[:-lag * FPS]
        feats["dtempo_%d" % lag] = d
        d2 = np.full(len(t1), np.nan)
        d2[:-lag * FPS] = t1[lag * FPS:] - t1[:-lag * FPS]
        feats["dtempo_fwd_%d" % lag] = d2
    with np.errstate(all="ignore"):
        mx = np.nanmax(P[:, :, 2].astype(np.float64), axis=1)
        q90 = np.nanpercentile(P[:, :, 2].astype(np.float64), 90, axis=1)
        px = P[:, :, 0].astype(np.float64) * L
        py = P[:, :, 1].astype(np.float64) * W
        spread = np.nanstd(px, axis=1) + np.nanstd(py, axis=1)
    feats["maxspd_1"] = rolling_mean(mx, FPS)
    feats["q90spd_1"] = rolling_mean(q90, FPS)
    feats["spread"] = rolling_mean(spread, FPS)
    s_f = seen.astype(float)
    for s in (1, 3, 8):
        feats["seen_%g" % s] = rolling_mean(s_f, int(s * FPS))
    since, until = _since_until(seen)
    feats["since"], feats["until"] = since, until
    bx = M["B"][:, 0].astype(np.float64) * L
    by = M["B"][:, 1].astype(np.float64) * W
    lx, ly = _ffill(bx, seen), _ffill(by, seen)
    nx, ny = _bfill(bx, seen), _bfill(by, seen)
    feats["last_touch_d"], feats["last_goal_d"] = line_distance(lx, ly, L, W)
    feats["next_touch_d"], feats["next_goal_d"] = line_distance(nx, ny, L, W)
    outside = np.nan_to_num(F["outside"], nan=-5.0)
    feats["outside_1"] = rolling_mean((outside > 0.5).astype(float), FPS)
    feats["bspeed_1"] = rolling_mean(F["bspeed"], FPS)
    dmin = np.fmin(F["dH"], F["dA"])
    feats["dmin_1"] = rolling_mean(dmin, FPS)
    names = [k for k in feats if k not in drop]
    return np.column_stack([feats[k] for k in names]).astype(np.float32), names


def train(XY):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.concatenate([x for x, _ in XY])
    y = np.concatenate([y for _, y in XY])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=200, l2_regularization=1.0,
                                         random_state=0)
    clf.fit(X, y)
    return clf


def training_rows(M, F, gt, step=3, drop=()):
    X, _ = features(M, F, drop)
    g = np.flatnonzero(gt != "X")[::step]
    return X[g], (gt[g] == "D").astype(np.int8)
