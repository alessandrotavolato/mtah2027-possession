"""Per-frame features from a match (see match.py). No labels are read here.

Everything is in metres and seconds.
"""
import numpy as np


def interpolate_ball(bx, by, max_gap):
    """Linearly interpolate ball gaps of at most `max_gap` frames."""
    seen = ~np.isnan(bx)
    x, y = bx.copy(), by.copy()
    idx = np.flatnonzero(seen)
    if len(idx) < 2:
        return x, y, seen
    gaps = np.diff(idx)
    for a, g in zip(idx[:-1][gaps > 1], gaps[gaps > 1]):
        if g - 1 <= max_gap:
            b = a + g
            t = np.arange(1, g) / g
            x[a + 1:b] = bx[a] + t * (bx[b] - bx[a])
            y[a + 1:b] = by[a] + t * (by[b] - by[a])
    return x, y, seen


def rolling_mean(v, half):
    """Centred nan-aware rolling mean over +/- half frames."""
    v = np.asarray(v, dtype=np.float64)
    ok = ~np.isnan(v)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(v))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(v), i + half + 1)
    cnt = k[hi] - k[lo]
    out = np.full(len(v), np.nan)
    m = cnt > 0
    out[m] = (c[hi] - c[lo])[m] / cnt[m]
    return out


def half_of_frame(M):
    h = np.zeros(len(M["B"]), dtype=np.int8)
    for half, start, stop in M["blocks"]:
        h[start:stop + 1] = half
    return h


def compute(M, interp_gap_s=1.0):
    P, B, team = M["P"], M["B"], M["team"]
    L, W, fps = M["length"], M["width"], M["fps"]
    n = len(B)
    px = P[:, :, 0].astype(np.float64) * L
    py = P[:, :, 1].astype(np.float64) * W
    pv = P[:, :, 2].astype(np.float64)

    bx_raw = B[:, 0].astype(np.float64) * L
    by_raw = B[:, 1].astype(np.float64) * W
    bx, by, seen = interpolate_ball(bx_raw, by_raw, int(interp_gap_s * fps))
    have = ~np.isnan(bx)

    d = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
    home = team == 0
    with np.errstate(all="ignore"):
        dH = np.nanmin(np.where(home[None, :], d, np.nan), axis=1)
        dA = np.nanmin(np.where(~home[None, :], d, np.nan), axis=1)

    vx = np.gradient(bx) * fps
    vy = np.gradient(by) * fps
    bspeed = rolling_mean(np.sqrt(vx ** 2 + vy ** 2), 2)

    with np.errstate(all="ignore"):
        mean_speed = np.nanmean(pv, axis=1)
    n_players = np.sum(~np.isnan(px), axis=1)

    out_x = np.maximum(-bx_raw, bx_raw - L)
    out_y = np.maximum(-by_raw, by_raw - W)
    outside = np.fmax(out_x, out_y)

    return dict(n=n, fps=fps, bx=bx, by=by, seen=seen, have=have, dH=dH, dA=dA,
                bspeed=bspeed, mean_speed=mean_speed, n_players=n_players, outside=outside)


def shape_features(M, smooth_half=12):
    """Ball-free team-shape features, direction-normalised so +x is towards the opponent goal."""
    P, team = M["P"], M["team"]
    L, W = M["length"], M["width"]
    half = half_of_frame(M)
    px = P[:, :, 0].astype(np.float64) * L
    py = P[:, :, 1].astype(np.float64) * W
    pv = P[:, :, 2].astype(np.float64)
    home = team == 0
    sign_home = np.where(half == 2, -M["sign1"], M["sign1"])

    cols, names = [], []
    with np.errstate(all="ignore"):
        for name, mask, sgn in (("H", home, sign_home), ("A", ~home, -sign_home)):
            x = np.where(mask[None, :], px, np.nan)
            y = np.where(mask[None, :], py, np.nan)
            v = np.where(mask[None, :], pv, np.nan)
            fwd = np.where(sgn[:, None] > 0, x, L - x)
            feats = {
                "cx": np.nanmean(fwd, 1), "cy": np.nanmean(y, 1),
                "sx": np.nanstd(fwd, 1), "sy": np.nanstd(y, 1),
                "max": np.nanmax(fwd, 1), "min": np.nanmin(fwd, 1),
                "q80": np.nanpercentile(fwd, 80, axis=1),
                "spd": np.nanmean(v, 1), "spdmax": np.nanmax(v, 1),
            }
            for k, val in feats.items():
                cols.append(val)
                names.append(name + "_" + k)
    X = np.column_stack(cols)
    idx = {nm: i for i, nm in enumerate(names)}
    extra = {
        "dcx": X[:, idx["H_cx"]] - (L - X[:, idx["A_cx"]]),
        "dspd": X[:, idx["H_spd"]] - X[:, idx["A_spd"]],
        "dsx": X[:, idx["H_sx"]] - X[:, idx["A_sx"]],
        "dsy": X[:, idx["H_sy"]] - X[:, idx["A_sy"]],
    }
    for k, val in extra.items():
        X = np.column_stack([X, val])
        names.append(k)
    for k in ("H_cx", "A_cx"):
        c = X[:, names.index(k)]
        vel = np.full(len(c), np.nan)
        vel[25:] = c[25:] - c[:-25]
        X = np.column_stack([X, vel])
        names.append(k + "_vel")
    Xs = np.column_stack([rolling_mean(X[:, j], smooth_half) for j in range(X.shape[1])])
    return Xs, names
