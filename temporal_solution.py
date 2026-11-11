"""Team-distance temporal model. Fixed assumptions; no possession labels used.

States H/A/X: distance evidence and optional automatic phase priors, decoded
with a transition penalty. Missing observations are uncertainty, not dead ball.
D is emitted only for a sustained observed position outside the pitch. This
boundary heuristic remains vulnerable to projection errors and aerial balls.
"""
from pathlib import Path
import numpy as np
from games import GAME, GAMES, require
from starter.baseline import team_of_each_track, team_names, write_submission
from starter.load_smart_tagging import load
from tag_solution import tag_masks


def features():
    meta, raw = require(['atd_meta', 'atd_raw'])
    teams = team_of_each_track(meta)
    n = GAMES[GAME]['n_video_frames']
    distance = np.full((n, 2), np.nan)
    outside = np.zeros(n, bool)
    pitch = np.array(GAMES[GAME]['pitch'])
    with open(raw) as f:
        for line in f:
            fid, players, ball = line.rstrip().split(':')
            i = int(fid) - 1
            b = np.array([float(x) for x in ball.split(',')[:2]])
            if not np.isfinite(b).all():
                continue
            p = np.array([[float(x) for x in g.split(',')[:2]] for g in players.split(';')])
            distances = np.linalg.norm((p - b) * pitch, axis=1)
            for t in (0, 1):
                d = distances[(teams == t) & np.isfinite(distances)]
                if len(d):
                    distance[i, t] = d.min()
            outside[i] = np.any(b * pitch < -0.5) or np.any((b - 1) * pitch > 0.5)
    masks = tag_masks(load(), n, team_names(meta))
    return distance, outside, masks


def decode(distance, outside, masks, use_tags=True):
    n = len(distance)
    seen = np.isfinite(distance).all(axis=1)
    # Distances become bounded log evidence; relative proximity dominates.
    score = np.zeros((n, 3))
    score[:, :2] = -0.3
    score[:, 2] = 0.0
    d = np.minimum(distance[seen], 20)
    logits = -d / 2.0
    logits -= np.max(logits, axis=1, keepdims=True)
    score[seen, :2] = logits
    score[seen, 2] = -2.0
    home, away, restart = masks
    if use_tags:
        # Conflicting phase tags cancel; restarts suppress the phase prior.
        score[:, 0] += 0.35 * (home & ~away & ~restart)
        score[:, 1] += 0.35 * (away & ~home & ~restart)
    # A short opposing glimpse must outweigh the transition cost.
    transitions = np.array([[0., -12., -6.], [-12., 0., -6.], [-6., -6., 0.]])
    back = np.zeros((n, 3), dtype=np.int8)
    value = score[0].copy()
    for i in range(1, n):
        candidates = value[:, None] + transitions
        back[i] = np.argmax(candidates, axis=0)
        value = candidates[back[i], np.arange(3)] + score[i]
    states = np.zeros(n, dtype=np.int8)
    states[-1] = np.argmax(value)
    for i in range(n-1, 0, -1):
        states[i-1] = back[i, states[i]]
    labels = np.array(['H','A','X'])[states]
    # Only sustained direct outside-pitch observations trigger D.
    edges = np.diff(np.r_[False, outside, False].astype(np.int8))
    for start,end in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)):
        if end-start >= 13:
            labels[start:end] = 'D'
    return labels


if __name__ == '__main__':
    distance, outside, masks = features()
    for use_tags, name in [(False, 'temporal_tracking'), (True, 'temporal_tags')]:
        labels = decode(distance, outside, masks, use_tags)
        path = Path(__file__).resolve().parent / 'evaluation/results' / (name + '.csv')
        write_submission(str(path), labels)
        print(name, dict(zip(*np.unique(labels, return_counts=True))))
