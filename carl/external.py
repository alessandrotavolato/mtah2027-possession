"""External validation games: Metrica's public sample data (github.com/metrica-sports/sample-data).

These are OTHER matches, with human-annotated event data -- allowed by RULES.md for training and
validation. We derive frame-level H/A/D labels from the events with mechanical rules:

  * dead:  from the end of the event that precedes each SET PIECE (ball out, shot off target,
           goal, foul...) until the set piece is taken; also before each period's kick-off.
  * owner: team of the most recent on-ball event (PASS, RECOVERY, SET PIECE, SHOT, BALL LOST,
           BALL OUT), switching at that event's start frame. Challenges are ignored (both teams).

Tracking for games 1-2 is CSV (one file per team). Player speed is derived from positions.
Set EXTERNAL_DIR to the sample-data checkout (default ~/code/external/sample-data).
"""
import os

import numpy as np
import pandas as pd

DIR = os.environ.get("EXTERNAL_DIR", os.path.expanduser("~/code/external/sample-data"))
FPS = 25
ON_BALL = {"PASS", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST", "BALL OUT"}


def _tracking(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    players = [c for c in cols if c.startswith("Player")]
    xy = []
    for c in players:
        i = cols.index(c)
        xy.append(raw.iloc[:, [i, i + 1]].values.astype(np.float32))
    bi = cols.index("Ball")
    ball = raw.iloc[:, [bi, bi + 1]].values.astype(np.float32)
    return raw["Period"].values, raw["Frame"].values, np.stack(xy, axis=1), ball


def _speed(xy, L, W):
    v = np.full(xy.shape[:2], np.nan, np.float32)
    dx = np.diff(xy[:, :, 0], axis=0) * L
    dy = np.diff(xy[:, :, 1], axis=0) * W
    s = np.sqrt(dx ** 2 + dy ** 2) * FPS
    v[1:] = s
    # light smoothing over +/- 5 frames
    from features import rolling_mean
    for j in range(v.shape[1]):
        v[:, j] = rolling_mean(v[:, j], 5)
    return v


def load_game(k, L=105.0, W=68.0):
    base = os.path.join(DIR, "data", "Sample_Game_{}".format(k))
    per, frame, h, ball = _tracking(os.path.join(base, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(k)))
    _, _, a, _ = _tracking(os.path.join(base, "Sample_Game_{}_RawTrackingData_Away_Team.csv".format(k)))
    xy = np.concatenate([h, a], axis=1)
    n = len(frame)
    P = np.full((n, xy.shape[1], 3), np.nan, np.float32)
    P[:, :, :2] = xy
    P[:, :, 2] = _speed(xy, L, W)
    B = np.full((n, 3), np.nan, np.float32)
    B[:, :2] = ball
    team = np.r_[np.zeros(h.shape[1], np.int8), np.ones(a.shape[1], np.int8)]

    ev = pd.read_csv(os.path.join(base, "Sample_Game_{}_RawEventsData.csv".format(k)))
    ev["s"] = ev["Start Frame"].astype(int) - 1     # tracking Frame counts from 1
    ev["e"] = ev["End Frame"].astype(int) - 1
    labels = np.full(n, "X", dtype="<U1")
    blocks = []
    for half in (1, 2):
        idx = np.flatnonzero(per == half)
        e = ev[ev.Period == half].sort_values("s")
        start, stop = int(e.s.min()), int(min(idx.max(), e.e.max()))
        # graded from the first kick-off to the last event of the period
        blocks.append((half, start, stop))
        owner = "D"
        rows = e.to_dict("records")
        seg = np.full(stop - start + 1, "D", dtype="<U1")
        for r in rows:
            if r["Type"] in ON_BALL and r["Team"] in ("Home", "Away"):
                s = max(r["s"], start)
                seg[s - start:] = "H" if r["Team"] == "Home" else "A"
        # dead spells before each set piece
        for i, r in enumerate(rows):
            if r["Type"] == "SET PIECE" and i > 0:
                prev = rows[i - 1]
                d0 = max(prev["e"], start)
                if r["s"] > d0:
                    seg[d0 - start:r["s"] - start] = "D"
        labels[start:stop + 1] = seg
    # attack direction: home GK (slot with mean x at extremes) side in first half
    first = per == 1
    with np.errstate(all="ignore"):
        hx = np.nanmean(h[first][:, :, 0])
    sign1 = 1.0 if hx < 0.5 else -1.0
    M = dict(P=P, B=B, team=team, blocks=blocks, sign1=sign1, length=L, width=W, fps=FPS)
    return M, labels
