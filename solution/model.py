"""Ball possession by Viterbi decoding over three states: H, A, D.

Per frame we compute evidence (log-scores) for each state from the tracking:
  - H vs A: which team has a player close to the ball, but only when somebody is
    close enough to be touching it (a ball in flight says nothing).
  - D: ball outside the pitch lines, players nearly stationary, ball missing.
Then a Viterbi pass finds the single best label sequence, paying a fixed cost
for every change of state. That cost is what turns flickering per-frame guesses
into whole possessions, and because Viterbi looks both ways in time the
changes land on the frame where the evidence flips, not a window later.

Finally the boundaries of each dead-ball stretch are snapped to physical
events: the restart is the frame the ball is kicked, and the stoppage is the
frame the ball crosses a line, when one is found nearby.
"""
import os
import sys

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from games import GAME, GAMES, half_video_blocks  # noqa: E402
from load import load  # noqa: E402

FPS = 25
L, W = GAMES[GAME]["pitch"]
D_, H_, A_ = 0, 1, 2
LETTERS = np.array(["D", "H", "A"])

PARAMS = dict(
    max_gap=25,          # interpolate ball gaps up to 1 s
    touch_radius=2.0,    # a player this close (m) can be touching the ball
    fade_radius=5.0,     # beyond this nobody is touching it
    team_weight=0.5,     # evidence per metre of distance advantage, per frame
    team_clip=4.0,
    out_margin=0.5,      # metres beyond a line before the ball counts as out
    out_score=1.5,
    still_speed=1.2,     # median player speed (m/s) below which play looks stopped
    speed_weight=1.5,
    missing_score=0.3,   # ball not tracked at all: weak hint of a stoppage
    moving_ball=-1.0,    # ball tracked, in bounds and kicked: play is on
    kick_speed=7.0,      # m/s; a carried or rolled-back ball is slower than a kick or throw
    pending_score=2.5,   # ball has gone out and not been restarted yet
    max_pending=FPS * 120,
    tag_pre=FPS * 5,     # Smart Tagging: frames before a set-piece tag that look dead
    tag_dead=1.0,
    tag_post=FPS * 3,    # frames after it credited to the tagged team
    tag_team=0.5,
    control_speed_max=12.0,  # m/s; above this the ball is passing, not controlled
    control_speed_ramp=6.0,
    cost_switch=16.0,    # price of an H<->A change
    cost_dead=25.0,      # price of entering or leaving D
    snap_window=75,      # frames to look for a kick / line crossing around a D edge
)


def interpolate_ball(ball, max_gap):
    """Fill gaps of at most max_gap frames linearly; longer gaps stay NaN."""
    out = ball[:, :2].copy()
    seen = ~np.isnan(out[:, 0])
    idx = np.where(seen)[0]
    if len(idx) < 2:
        return out
    gaps = np.diff(idx)
    for k in np.where((gaps > 1) & (gaps <= max_gap))[0]:
        a, b = idx[k], idx[k + 1]
        t = (np.arange(a + 1, b) - a) / (b - a)
        out[a + 1:b] = out[a] + t[:, None] * (out[b] - out[a])
    return out


def rolling_mean(x, half):
    x = np.where(np.isnan(x), np.nanmedian(x), x)
    c = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half)
    hi = np.minimum(len(x), i + half + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def features(players, ball, team, p):
    bxy = interpolate_ball(ball, p["max_gap"])
    seen = ~np.isnan(bxy[:, 0])
    dist = np.hypot(players[:, :, 0] - bxy[:, None, 0], players[:, :, 1] - bxy[:, None, 1])
    dist = np.where(np.isnan(dist), np.inf, dist)
    d_home = dist[:, team == 0].min(1)
    d_away = dist[:, team == 1].min(1)
    speed = np.r_[0.0, np.hypot(*np.diff(bxy, axis=0).T) * FPS]
    speed = np.where(np.isnan(speed), 0.0, speed)
    with np.errstate(all="ignore"):
        pspeed = np.nanmedian(players[:, :, 2], axis=1)
    m = p["out_margin"]
    out = seen & ((bxy[:, 0] < -m) | (bxy[:, 0] > L + m) | (bxy[:, 1] < -m) | (bxy[:, 1] > W + m))
    kick = seen & ~out & (speed > p["kick_speed"]) & (speed - np.r_[np.zeros(5), speed[:-5]] > 3.0)
    return dict(bxy=bxy, seen=seen, d_home=d_home, d_away=d_away, ball_speed=speed,
                pspeed=rolling_mean(pspeed, FPS // 2), out=out, kick=kick,
                pending=pending_restart(bxy, seen, out, kick, p))


def pending_restart(bxy, seen, out, kick, p):
    """True from the moment the ball leaves the pitch until the restart.

    The laws of the game, applied to the tracking: once the ball is out it stays dead until a
    throw-in, goal kick or corner puts it back, and those restarts are a kick or throw -- a
    tracked, in-bounds ball that suddenly accelerates. Carrying or rolling the ball back to the
    spot does not end the stoppage. The ball vanishing near a line (it often leaves the tactical
    camera's view as it goes out) counts as leaving.
    """
    n = len(seen)
    near_line = seen & ((bxy[:, 0] < 1.0) | (bxy[:, 0] > L - 1.0) | (bxy[:, 1] < 1.0) | (bxy[:, 1] > W - 1.0))
    pending = np.zeros(n, dtype=bool)
    active, since, last_near = False, 0, -10 ** 9
    for t in range(n):
        if near_line[t]:
            last_near = t
        starts = out[t] or (not seen[t] and t - last_near < 3 and t > 0 and seen[t - 1])
        if starts and not active:
            active, since = True, t
        elif active and (kick[t] or t - since > p["max_pending"]):
            active = False
        pending[t] = active
    return pending


def emissions(f, p, tags=None):
    n = len(f["seen"])
    e = np.zeros((n, 3))
    # Team evidence: distance advantage, faded out when nobody is near the ball.
    near = np.minimum(f["d_home"], f["d_away"])
    fade = np.clip((p["fade_radius"] - near) / (p["fade_radius"] - p["touch_radius"]), 0, 1)
    adv = np.clip(f["d_away"] - f["d_home"], -p["team_clip"], p["team_clip"])
    adv = np.where(np.isfinite(adv), adv, 0.0)
    # A ball flying past a player at pace is not under his control: only a ball slow enough to
    # be received or carried counts, which is how "controlled possession" is defined.
    control = np.clip((p["control_speed_max"] - f["ball_speed"]) / p["control_speed_ramp"], 0, 1)
    s = p["team_weight"] * adv * fade * control
    e[:, H_] = s / 2
    e[:, A_] = -s / 2
    # Dead-ball evidence.
    dead = p["speed_weight"] * np.clip(p["still_speed"] - f["pspeed"], -1.5, 1.0) / 0.4
    dead = np.clip(dead, -3.0, 2.0)
    dead += np.where(f["out"], p["out_score"], 0.0)
    dead += np.where(~f["seen"], p["missing_score"], 0.0)
    live_ball = f["seen"] & ~f["out"] & ~f["pending"] & (f["ball_speed"] > p["kick_speed"])
    dead += np.where(live_ball, p["moving_ball"], 0.0)
    dead += np.where(f["pending"], p["pending_score"], 0.0)
    e[:, D_] = dead
    if tags is not None:
        e += tags
    return e


def tag_evidence(n, p):
    """Smart Tagging priors: a restart happens near the start of each set-piece tag, and the
    tagged team takes it. Kept weak -- the tagging is automatic and has its own errors."""
    sys.path.insert(0, os.path.join(ROOT, "starter"))
    import load_smart_tagging as st
    e = np.zeros((n, 3))
    team_state = {"Barcelona": H_, "Atletic Club": A_}
    for ev in st.load():
        if ev["code"] != "SET PIECES":
            continue
        s = ev["start_frame"]
        # Players set up for the restart just before the tag starts.
        e[max(0, s - p["tag_pre"]):s, D_] += p["tag_dead"]
        side = team_state.get(ev["tags"].get("Team"))
        if side is not None:
            e[s:s + p["tag_post"], side] += p["tag_team"]
    return e


def viterbi(e, p):
    n = len(e)
    cost = np.array([[0, p["cost_dead"], p["cost_dead"]],
                     [p["cost_dead"], 0, p["cost_switch"]],
                     [p["cost_dead"], p["cost_switch"], 0]])
    score = e[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] - cost          # from i to j
        back[t] = cand.argmax(0)
        score = cand.max(0) + e[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def runs(path):
    d = np.flatnonzero(np.diff(path)) + 1
    starts = np.r_[0, d]
    ends = np.r_[d, len(path)]
    return [(path[s], s, e) for s, e in zip(starts, ends)]


def snap_dead_edges(path, f, p):
    """Move each D stretch's end to the restart kick, and its start to the line crossing."""
    path = path.copy()
    w = p["snap_window"]
    seen, speed, out, near_team = f["seen"], f["ball_speed"], f["out"], None
    near_team = np.where(f["d_home"] <= f["d_away"], H_, A_)
    for state, s, e in runs(path):
        if state != D_ or s == 0 or e == len(path):
            continue
        # Restart: first kick in [e-w, e+w].
        lo, hi = max(s, e - w), min(len(path) - 1, e + w)
        hits = np.flatnonzero(f["kick"][lo:hi])
        kick = lo + hits[0] if len(hits) else None
        if kick is not None and kick > s:
            new_team = near_team[kick]
            if kick < e:
                path[kick:e] = new_team
            else:
                path[e:kick] = D_
        # Stoppage: ball crossing a line shortly before or after the D start.
        lo, hi = max(0, s - w), min(e, s + w)
        crossing = np.flatnonzero(out[lo:hi])
        if len(crossing):
            c = lo + crossing[0]
            if c < s:
                path[c:s] = D_
            elif c > s:
                path[s:c] = path[s - 1]
    return path


def predict(p=PARAMS):
    players, ball, team = load()
    n = len(ball)
    f = features(players, ball, team, p)
    e = emissions(f, p, tag_evidence(n, p))
    labels = np.full(n, "X", dtype="<U1")
    for _, first, last in half_video_blocks(GAME):
        sl = slice(first, last + 1)
        sub = {k: v[sl] for k, v in f.items()}
        path = viterbi(e[sl], p)
        path = snap_dead_edges(path, sub, p)
        labels[sl] = LETTERS[path]
    return labels


def write(labels, path):
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def summary(labels):
    graded = labels != "X"
    lab = labels[graded]
    poss = [r for r in runs(np.searchsorted(["A", "D", "H", "X"], labels)) if labels[r[1]] in "HA"]
    durs = np.array([(e - s) / FPS for _, s, e in poss])
    return ("H {:.1%}  A {:.1%}  D {:.1%}   possessions {}  mean {:.1f}s  median {:.1f}s"
            .format(np.mean(lab == "H"), np.mean(lab == "A"), np.mean(lab == "D"),
                    len(durs), durs.mean(), np.median(durs)))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "evaluation", "results", "viterbi_v1.csv")
    labels = predict()
    write(labels, out)
    print(summary(labels))
    print("wrote", out)
