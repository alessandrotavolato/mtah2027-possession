"""Possession segmentation: per-frame evidence + Viterbi over {H, A, D}, in two passes.

Idea
----
A possession is an interval, so we decode the most likely *segmentation* rather than labelling
frames independently. Each frame gets a score for H, A and D; switching state costs a fixed
penalty. The Viterbi path maximises total score minus penalties, so a switch only happens when the
evidence for the new state, accumulated over time, outweighs the penalty. Frames with no evidence
keep the current state -- "a loose ball stays with the last team in possession".

Pass 1 -- ball evidence (tactical ATD only):
  * control:  nearest player within CONTROL_R m of the ball and clearly closer than the nearest
              opponent -> that team, stronger the closer he is.
  * outside:  ball observed beyond the touch/goal line -> dead.
  * tempo:    players slow down collectively in a stoppage -> dead; high tempo -> alive.
  * no-ball:  ball unseen for a while and play slow -> weak dead.
  * kick:     the ball accelerating away from a player after a slow phase -> play (re)starts.

Pass 2 -- self-supervised team-shape model:
  The ball is missing on ~41% of graded frames, almost always in blocks of several seconds, so
  ball evidence alone cannot say who has it there. We train a classifier on the frames where the
  ball IS visible, using pass-1's own decoded labels as targets (pseudo-labels; no human labels)
  and only ball-free features (team centroids, spreads, speeds, forward motion). Its log-probabilities
  are then added as evidence on the ball-missing frames, and the match is decoded again.

All parameters are football priors (metres, seconds, m/s) fixed in advance, not fitted to labels.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from features import compute, rolling_mean, shape_features, half_of_frame  # noqa: E402

FPS = 25

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

PARAMS = dict(
    interp_gap_s=1.0,      # bridge ball gaps up to 1 s by linear interpolation
    control_r=2.0,         # m: a player this close to the ball can be controlling it
    margin_full=1.0,       # m: margin over nearest opponent for full-strength evidence
    w_team=1.0,            # score per frame of clear control
    w_out=3.0,             # score per frame of ball clearly outside the pitch
    out_margin=0.75,       # m outside the line before we believe it (tracking noise)
    tempo_half_s=1.0,      # smoothing of mean player speed
    tempo_dead=1.3,        # m/s: below this, players are standing -> stoppage
    tempo_alive=2.0,       # m/s: above this, play is on
    w_tempo=0.6,
    noball_s=2.0,          # ball unseen this long on both sides -> evidence of dead
    w_noball=0.3,
    kick_speed=7.0,        # m/s: ball speed of a pass/shot
    kick_from_r=1.5,       # m: ... starting from a player this close
    kick_after_s=1.0,      # alive evidence lasts this long after the kick
    w_kick=1.5,
    pen_switch=12.0,       # H<->A switch penalty (~0.5 s of clear control)
    pen_dead=30.0,         # alive<->dead penalty
    # pass 2
    use_shape=False,
    w_shape=0.5,           # weight of classifier log-probs on ball-missing frames
    shape_margin_s=1.0,    # train only on frames this far from a pass-1 boundary
    # team assignment inside alive spells
    team_rule="touch",     # "viterbi" | "touch" | "hybrid"
    touch_r=0.8,           # m: ball this close to a player = contact
    min_ctrl_s=0.4,        # s: contact this long = control
    next_s=3.0,            # s: or the next contact within this time is by a teammate
    window_s=0.0,          # s: cumulative-contact window for control under pressure
    ratio=1.0,             # new team needs > ratio x the old team's contact time in it
    merge_gap=3,           # frames: same-team contacts this close are one contact
    touch_switch=True,     # a brief contact that changes the ball's course also switches
    angle_deg=35.0,        # ... course change of at least this angle
    speed_ratio=0.5,        # ... or this relative change of ball speed
    hybrid_gap_s=2.0,      # s: ball missing on +/- this window -> use shape-informed decode
    # external dead-ball classifier
    w_dead=0.5,            # weight on its per-frame log-odds
    dead_no_visibility=False,  # drop ball-visibility features (domain gap: see README)
    w_out_clf=3.0,         # keep the hard "ball outside the pitch" rule alongside the classifier
    dead_bias=-1.5,        # log-odds offset (0 = classifier's own calibration)
)


def ball_evidence(F, p):
    n = F["n"]
    S = np.zeros((n, 3))

    # --- team control -------------------------------------------------------------------------
    dH, dA, have = F["dH"], F["dA"], F["have"]
    ok = have & ~np.isnan(dH) & ~np.isnan(dA)
    dmin = np.where(ok, np.fmin(dH, dA), np.inf)
    margin = np.abs(dH - dA)
    close = np.clip(1.0 - dmin / p["control_r"], 0, 1)  # 1 at the ball, 0 at control_r
    strength = np.where(ok, np.sqrt(close) * np.clip(margin / p["margin_full"], 0, 1), 0.0)
    homeclose = dH < dA
    S[:, H] += np.where(homeclose, strength, 0) * p["w_team"]
    S[:, A] += np.where(~homeclose, strength, 0) * p["w_team"]

    # --- ball outside the pitch ---------------------------------------------------------------
    outside = np.nan_to_num(F["outside"], nan=-1.0)
    S[:, D] += np.where(outside > p["out_margin"], p["w_out"], 0.0)

    # --- collective tempo ---------------------------------------------------------------------
    tempo = np.nan_to_num(rolling_mean(F["mean_speed"], int(p["tempo_half_s"] * FPS)), nan=0.0)
    lo, hi = p["tempo_dead"], p["tempo_alive"]
    t = np.clip((hi - tempo) / (hi - lo), 0, 1) * 2 - 1   # +1 dead ... -1 alive
    S[:, D] += p["w_tempo"] * t

    # --- ball unseen for a long time ----------------------------------------------------------
    recent = rolling_mean(F["seen"].astype(float), int(p["noball_s"] * FPS))
    S[:, D] += np.where(recent == 0, p["w_noball"], 0.0)

    # --- kick: ball leaves a player fast -> play is alive right after -------------------------
    bs = np.nan_to_num(F["bspeed"], nan=0.0)
    near = np.nan_to_num(dmin, posinf=99) < p["kick_from_r"]
    lag = int(0.4 * FPS)
    near_before = np.zeros(n, bool)
    for k in range(1, lag + 1):
        near_before[k:] |= near[:-k]
    kick = (bs > p["kick_speed"]) & near_before & (np.r_[0, bs[:-1]] <= p["kick_speed"])
    after = np.zeros(n)
    for f in np.flatnonzero(kick):
        after[f:f + int(p["kick_after_s"] * FPS)] = 1.0
    S[:, D] -= p["w_kick"] * after
    return S


def viterbi(S, p):
    n = len(S)
    T = np.array([[0, p["pen_switch"], p["pen_dead"]],
                  [p["pen_switch"], 0, p["pen_dead"]],
                  [p["pen_dead"], p["pen_dead"], 0]], dtype=float)
    score = S[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    cols = np.arange(3)
    for t in range(1, n):
        cand = score[:, None] - T
        b = np.argmax(cand, axis=0)
        back[t] = b
        score = cand[b, cols] + S[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = np.argmax(score)
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(S, p, blocks):
    codes = np.full(len(S), -1, dtype=np.int8)
    for _, start, stop in blocks:
        codes[start:stop + 1] = viterbi(S[start:stop + 1], p)
    return codes


def stable_mask(codes, margin):
    """Frames at least `margin` frames away from any change of decoded state."""
    change = np.r_[False, codes[1:] != codes[:-1]]
    dist = np.full(len(codes), 10 ** 9)
    last = -10 ** 9
    for i in range(len(codes)):
        if change[i]:
            last = i
        dist[i] = i - last
    last = 10 ** 9
    for i in range(len(codes) - 1, -1, -1):
        if i + 1 < len(codes) and change[i + 1]:
            last = i + 1
        dist[i] = min(dist[i], last - i)
    return dist >= margin


def shape_logprob(codes, F, M, half, p, seed=0):
    """Self-supervised classifier: ball-free team shape -> pass-1 state. Log-probs per frame."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X, _ = shape_features(M)
    graded = half > 0
    train = graded & F["seen"] & (codes >= 0) & stable_mask(codes, int(p["shape_margin_s"] * FPS))
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         l2_regularization=1.0, random_state=seed)
    # subsample every 5th frame: consecutive frames are near-duplicates
    idx = np.flatnonzero(train)[::5]
    clf.fit(X[idx], codes[idx])
    proba = np.full((len(codes), 3), 1.0 / 3)
    proba[graded] = clf.predict_proba(X[graded])
    return np.log(np.clip(proba, 1e-3, 1.0))


def dead_evidence(M, F, clf, p):
    """Replace the hand-made dead/alive terms by the external classifier's log-odds."""
    from dead_model import features, VISIBILITY
    X, _ = features(M, F, VISIBILITY if p["dead_no_visibility"] else ())
    pd = np.clip(clf.predict_proba(X)[:, 1], 1e-3, 1 - 1e-3)
    return p["w_dead"] * (np.log(pd) - np.log(1 - pd) - p["dead_bias"])


def predict(M, p=None, F=None, dead_clf=None):
    p = dict(PARAMS, **(p or {}))
    if F is None:
        F = compute(M, p["interp_gap_s"])
    n = F["n"]
    half = half_of_frame(M)
    if dead_clf is not None:
        S = ball_evidence(F, dict(p, w_out=p["w_out_clf"], w_tempo=0, w_noball=0, w_kick=0))
        S[:, D] += dead_evidence(M, F, dead_clf, p)
    else:
        S = ball_evidence(F, p)
    codes = decode(S, p, M["blocks"])
    if p["use_shape"]:
        L = shape_logprob(codes, F, M, half, p)
        L = L - L.mean(axis=1, keepdims=True)
        missing = ~F["have"]
        S2 = S.copy()
        S2[missing] += p["w_shape"] * L[missing]
        codes = decode(S2, p, M["blocks"])
    if p["team_rule"] in ("touch", "hybrid"):
        from touches import possession
        alive = (codes == H) | (codes == A)
        team = possession(alive, F["dH"], F["dA"], F["have"], touch_r=p["touch_r"],
                          min_ctrl_s=p["min_ctrl_s"], next_s=p["next_s"], prior=codes,
                          window_s=p["window_s"], ratio=p["ratio"], merge_gap=p["merge_gap"],
                          bx=F["bx"], by=F["by"], touch_switch=p["touch_switch"],
                          angle_deg=p["angle_deg"], speed_ratio=p["speed_ratio"])
        if p["team_rule"] == "hybrid":
            # inside long ball-missing stretches trust the shape-informed decode
            gap = rolling_mean((~F["have"]).astype(float), int(p["hybrid_gap_s"] * FPS)) >= 1.0
            team = np.where(gap & alive, codes, team)
        graded = codes >= 0
        codes = np.where(graded, np.where(alive, team, D), codes).astype(np.int8)
    labels = np.full(n, "X", dtype="<U1")
    g = codes >= 0
    labels[g] = LETTERS[codes[g]]
    return labels
