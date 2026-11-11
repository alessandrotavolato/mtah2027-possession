"""Experiment: bridge missing tracking using agreeing tactical phase evidence.

An additional long gap is bridged only when both neighbouring tracking observations and an
attacking-phase tag agree on the team for the entire gap. Any opposing phase
or set-piece overlap vetoes it. Tags are automatic and may be wrong; they are
not treated as exact possession or stoppage boundaries. No labels are used.
"""
from pathlib import Path
import numpy as np
from games import GAME, GAMES
from starter.baseline import build, smooth, write_submission, NOBODY, team_names
from starter.load_smart_tagging import load
from games import require
from first_solution import bridge_gaps

# Fixed assumptions, not fitted against this match's public labels.
PHASES = {'BUILD UP', 'PROGRESSION', 'CREATING CHANCES', 'ATTACKING TRANSITION', 'LONG BALL'}
MAX_GAP_SECONDS = 30
SMOOTHING_HALF_WINDOW = 50
OUTPUT = Path(__file__).resolve().parent / 'evaluation/results/tag_solution.csv'


def tag_masks(events, size, names):
    home = np.zeros(size, dtype=bool)
    away = np.zeros(size, dtype=bool)
    restart = np.zeros(size, dtype=bool)
    teams = {names[0].casefold(): home, names[1].casefold(): away}
    for event in events:
        start = max(0, min(size, event['start_frame']))
        end = max(start, min(size, event['end_frame']))
        code = (event['code'] or '').strip().upper()
        if code == 'SET PIECES':
            restart[start:end] = True
        elif code in PHASES:
            team = (event['tags'].get('Team') or '').strip().casefold()
            if team in teams:
                teams[team][start:end] = True
    return home, away, restart


def apply_tags(raw, initial, masks, max_gap):
    result = initial.copy()
    home, away, restart = masks
    missing = raw == NOBODY
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    runs = frames = 0
    for start, end in zip(starts, ends):
        if start == 0 or end == len(raw) or end - start > max_gap:
            continue
        team = raw[start - 1]
        if team != raw[end] or np.any(restart[start:end]):
            continue
        own, other = (home, away) if team == 0 else (away, home)
        if np.all(own[start:end]) and not np.any(other[start:end]):
            changed = np.count_nonzero(result[start:end] == NOBODY)
            if changed:
                result[start:end] = team
                runs += 1
                frames += changed
    return result, runs, frames


if __name__ == '__main__':
    raw = build()
    names = team_names(require(['atd_meta'], GAME)[0])
    masks = tag_masks(load(), len(raw), names)
    initial, _, _ = bridge_gaps(raw, 5 * GAMES[GAME]['fps'])
    # Preserve original observations inside restart windows; earlier bridging must
    # not bypass the restart veto used for the additional tag-supported rule.
    initial[masks[2]] = raw[masks[2]]
    codes, runs, frames = apply_tags(raw, initial, masks, MAX_GAP_SECONDS * GAMES[GAME]['fps'])
    # Smooth independently across restart-context boundaries, so a vote from
    # outside a restart window cannot leak across its edges. A restart window
    # itself is not assumed to be entirely dead ball.
    cuts = np.r_[0, np.flatnonzero(np.diff(masks[2].astype(int))) + 1, len(raw)]
    labels = np.empty(len(raw), dtype='<U1')
    for start, end in zip(cuts[:-1], cuts[1:]):
        labels[start:end] = smooth(codes[start:end], SMOOTHING_HALF_WINDOW)
    write_submission(str(OUTPUT), labels)
    print('Tags supported {} additional gaps ({} frames).'.format(runs, frames))
    print('Saved {}'.format(OUTPUT))
