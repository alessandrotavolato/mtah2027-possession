"""Self-supervised test: hide observed tracking and try to recover nearest teams.

This measures agreement with noisy tracking, NOT real possession or spellF1.
Artificial gaps start in fully observed periods, so may differ from real gaps.
No ground-truth labels are read. Fixed seed makes the experiment repeatable.
"""
import numpy as np
from starter.baseline import build, smooth, NOBODY
from first_solution import bridge_gaps
from tag_solution import tag_masks, apply_tags
from starter.load_smart_tagging import load
from starter.baseline import team_names
from games import GAME, GAMES, require


def test(raw, masks, width, seed=7):
    rng = np.random.default_rng(seed)
    seen = raw != NOBODY
    bad = np.r_[0, np.cumsum(~seen)]
    # Require the entire interval and both adjacent frames to be observed.
    starts = np.arange(1, len(raw) - width)
    starts = starts[(bad[starts + width + 1] - bad[starts - 1]) == 0]
    rng.shuffle(starts)
    occupied = np.zeros(len(raw), bool)
    selected = []
    for start in starts:
        # Keep held-out windows apart so they cannot merge or affect one another.
        lo, hi = max(0, start - 125), min(len(raw), start + width + 125)
        if not occupied[lo:hi].any():
            selected.append(start)
            occupied[start:start + width] = True
        if len(selected) == 100:
            break
    masked = raw.copy()
    masked[occupied] = NOBODY
    bridged, _, _ = bridge_gaps(masked, 5 * GAMES[GAME]['fps'])
    tagged, _, _ = apply_tags(masked, bridged, masks, 30 * GAMES[GAME]['fps'])
    target = np.where(raw[occupied] == 0, 'H', 'A')
    print('{} s gaps: {} windows, {} hidden frames'.format(width / GAMES[GAME]['fps'], len(selected), len(target)))
    for name, codes in [('baseline smoothing', masked), ('short gap bridge', bridged), ('tag-supported bridge', tagged)]:
        prediction = smooth(codes, 50)[occupied]
        print('  {:22s} agreement {:.1%}; dead/unknown {:.1%}'.format(name, np.mean(prediction == target), np.mean(prediction == 'D')))


if __name__ == '__main__':
    raw = build()
    names = team_names(require(['atd_meta'], GAME)[0])
    masks = tag_masks(load(), len(raw), names)
    for seconds in (1, 3, 6):
        test(raw, masks, seconds * GAMES[GAME]['fps'])
