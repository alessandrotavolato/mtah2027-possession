"""Experimental complete H/A/D/X pipeline, with fixed untuned assumptions.

Uses bounded relative-distance evidence, a centered majority window, short
interpolation, and a minimum-duration opposing-evidence requirement. Missing
ball positions mean uncertainty, not dead ball. A conservative boundary rule
is the only source of D, so many real stoppages will remain unresolved.
No possession labels are loaded. Dependencies: requirements-tested.txt.
"""
import argparse
from pathlib import Path
import numpy as np
from temporal_solution import features
from starter.baseline import write_submission

FPS = 25
MAX_INTERPOLATION = 3 * FPS
VOTE_HALF_WINDOW = 2 * FPS
MINIMUM_SWITCH_SUPPORT = 13  # approximately half a second
MINIMUM_OUTSIDE_DURATION = 13


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def rolling_sum(values, half):
    cumulative = np.r_[0., np.cumsum(values)]
    index = np.arange(len(values))
    return cumulative[np.minimum(len(values), index + half + 1)] - cumulative[np.maximum(0, index - half)]


def predict(distance, outside, masks, use_tags=False, fps=25):
    n = len(distance)
    max_interpolation = round(3 * fps)
    half_window = round(2 * fps)
    minimum_support = max(1, np.ceil(0.5 * fps))
    if n == 0:
        return np.empty(0, dtype='<U1')
    seen = np.isfinite(distance).all(axis=1)
    home, away, restart = masks
    d = distance.copy()
    interpolated = np.zeros(n, bool)
    # Never infer continuous live play across a restart-context window or an
    # outside-pitch observation; long natural gaps remain unresolved.
    for start, end in runs(~seen):
        if start == 0 or end == n or end-start > max_interpolation:
            continue
        if np.any(restart[start:end]) or np.any(outside[start-1:end+1]):
            continue
        for team in (0, 1):
            d[start:end, team] = np.linspace(d[start-1, team], d[end, team], end-start+2)[1:-1]
        interpolated[start:end] = True
    available = seen | interpolated
    strength = np.zeros(n)
    strength[available] = np.tanh((d[available, 1] - d[available, 0]) / 2.)
    # Interpolation carries weaker evidence than actual detections.
    weight = seen.astype(float) + 0.5 * interpolated.astype(float)
    strength *= weight
    if use_tags:
        prior = (home & ~away & ~restart).astype(float) - (away & ~home & ~restart).astype(float)
        # The phase prior can disambiguate nearby tracking, not conjure evidence
        # during an entirely unobserved long interval.
        strength += 0.1 * prior * available
    # Restart contexts partition the smoothing window, without labelling entire
    # restart windows as dead ball or treating their edges as precise whistles.
    cuts = np.r_[0, np.flatnonzero(np.diff(restart.astype(int))) + 1, n]
    labels = np.full(n, 'X', dtype='<U1')
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        vote = rolling_sum(strength[lo:hi], half_window)
        support = rolling_sum(weight[lo:hi], half_window)
        proposed = np.where(vote >= 0, 'H', 'A')
        # At least half a second of effective observations must support a vote.
        reliable = support >= minimum_support
        proposed[~reliable] = 'X'
        # Suppress brief opposing runs; abstention gaps reset continuity.
        current = 'X'
        pending = None
        count = 0
        start = 0
        result = proposed.copy()
        for i, candidate in enumerate(proposed):
            if candidate == 'X':
                current, pending, count = 'X', None, 0
            elif current == 'X':
                current = candidate
            elif candidate == current:
                pending, count = None, 0
            else:
                if pending != candidate:
                    pending, count, start = candidate, 0, i
                count += 1
                result[i] = current
                if count >= minimum_support:
                    result[start:i+1] = candidate
                    current, pending, count = candidate, None, 0
        labels[lo:hi] = result
    # D requires a sustained *observed* outside-pitch position. Projection errors,
    # aerial balls and re-entry can still violate this heuristic.
    for start, end in runs(outside & seen):
        if end-start >= minimum_support:
            labels[start:end] = 'D'
    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tags', action='store_true')
    args = parser.parse_args()
    distance, outside, masks = features()
    labels = predict(distance, outside, masks, args.tags)
    name = 'hybrid_tags' if args.tags else 'hybrid_tracking'
    path = Path(__file__).resolve().parent / 'evaluation/results' / (name + '.csv')
    write_submission(str(path), labels)
    print(name, {label: int(np.sum(labels == label)) for label in 'HADX'})
