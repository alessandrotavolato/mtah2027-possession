"""Touch-sequence possession rule, following CHALLENGE.md's definition.

  * A *contact* is a run of frames where the ball is within TOUCH_R of a player; it belongs to
    the team of the nearest player (majority over the run).
  * Possession changes to team T at a contact by T only if the contact is *controlled*: T keeps
    the ball for at least MIN_CTRL_S, or the next contact (within NEXT_S) is also by T -- a
    completed reception or pass. A lone touch by the opponent followed by the original team
    getting it back is a deflection and does not change possession.
  * After a dead ball the first contact is the restart and always sets the possessor.
  * With no contacts (ball in flight, or ball not tracked) possession stays with the last team.

Dead/alive is decided separately (model.alive_mask) and passed in.
"""
import numpy as np

FPS = 25


def contacts(dH, dA, have, touch_r, merge_gap):
    """[(start, stop, team)] with team 0=H, 1=A."""
    ok = have & ~np.isnan(dH) & ~np.isnan(dA)
    dmin = np.where(ok, np.fmin(dH, dA), np.inf)
    team = np.where(dH < dA, 0, 1)
    c = dmin < touch_r
    out = []
    i, n = 0, len(c)
    while i < n:
        if not c[i]:
            i += 1
            continue
        j = i
        while j < n and c[j]:
            j += 1
        t = int(np.round(team[i:j].mean()))
        if out and out[-1][2] == t and i - out[-1][1] <= merge_gap:
            out[-1] = (out[-1][0], j, t)
        else:
            out.append((i, j, t))
        i = j
    return out


def deflected(bx, by, s, f, k=5, angle_deg=35.0, speed_ratio=0.35):
    """Did the ball's velocity change across contact [s, f)? (direction or speed)"""
    n = len(bx)
    a0, a1 = max(0, s - k), s
    b0, b1 = f, min(n - 1, f + k)
    if a1 - a0 < 2 or b1 - b0 < 2:
        return False
    v0 = np.array([bx[a1] - bx[a0], by[a1] - by[a0]]) / (a1 - a0)
    v1 = np.array([bx[b1] - bx[b0], by[b1] - by[b0]]) / (b1 - b0)
    if np.any(np.isnan(v0)) or np.any(np.isnan(v1)):
        return False
    s0, s1 = np.hypot(*v0), np.hypot(*v1)
    if max(s0, s1) < 0.04:          # ball (nearly) still: < 1 m/s, nothing to deflect
        return False
    cos = np.dot(v0, v1) / max(s0 * s1, 1e-9)
    turned = cos < np.cos(np.radians(angle_deg))
    changed = abs(s1 - s0) > speed_ratio * max(s0, s1)
    return bool(turned or changed)


def possession(alive, dH, dA, have, touch_r=1.2, merge_gap=3, min_ctrl_s=0.4, next_s=3.0,
               prior=None, window_s=0.0, ratio=1.0, bx=None, by=None, touch_switch=False,
               angle_deg=35.0, speed_ratio=0.35):
    """Per-frame codes 0/1/2 (H/A/D). `alive` bool array. `prior`: fallback team per frame
    (used only to seed an alive spell that has no contact at all)."""
    n = len(alive)
    codes = np.full(n, 2, dtype=np.int8)
    cl = contacts(dH, dA, have & alive, touch_r, merge_gap)
    min_ctrl = int(min_ctrl_s * FPS)
    next_gap = int(next_s * FPS)
    window = int(window_s * FPS)

    # alive spells
    e = np.diff(np.r_[0, alive.astype(np.int8), 0])
    starts, stops = np.flatnonzero(e == 1), np.flatnonzero(e == -1)
    ci = 0
    for a, b in zip(starts, stops):
        while ci < len(cl) and cl[ci][1] <= a:
            ci += 1
        k = ci
        spell = []
        while k < len(cl) and cl[k][0] < b:
            spell.append(cl[k])
            k += 1
        if not spell:
            t0 = int(prior[a]) if prior is not None and prior[a] in (0, 1) else 0
            codes[a:b] = t0
            continue
        cur = spell[0][2]          # restart taker
        change_at = [(a, cur)]
        for m in range(1, len(spell)):
            s, f, t = spell[m]
            if t == cur:
                continue
            controlled = (f - s) >= min_ctrl
            if not controlled and m + 1 < len(spell):
                s2, _, t2 = spell[m + 1]
                controlled = (t2 == t) and (s2 - f) <= next_gap
            if not controlled and window > 0:
                # under pressure the nearest player flickers: count contact time per team over
                # the next `window` frames instead of requiring one contiguous contact
                own = other = 0
                for s3, f3, t3 in spell[m:]:
                    if s3 >= s + window:
                        break
                    dur = min(f3, s + window) - s3
                    if t3 == t:
                        own += dur
                    else:
                        other += dur
                controlled = own >= min_ctrl and own > ratio * other
            if not controlled and touch_switch and bx is not None:
                # a brief but real touch (the ball changes course) by the other team
                controlled = deflected(bx, by, s, f, angle_deg=angle_deg, speed_ratio=speed_ratio)
            if controlled:
                cur = t
                change_at.append((s, t))
        for idx, (s, t) in enumerate(change_at):
            e2 = change_at[idx + 1][0] if idx + 1 < len(change_at) else b
            codes[s:e2] = t
    return codes
