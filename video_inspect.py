"""Overlay provided tracking on one tactical-video frame for pipeline debugging.

Requires ffmpeg and the original match video. Run locally; do not publish frames
or derived tracking. Video inspection does not create labels or tune on this match.
"""
import argparse
import csv
import math
import os
import subprocess
from games import GAME, GAMES, require
from starter.baseline import team_of_each_track
from starter.load_homography import project


def row_at(path, frame):
    with open(path) as f:
        for line in f:
            fid, players, ball = line.rstrip().split(':')
            if int(fid) == frame + 1:
                return players.split(';'), ball
    raise ValueError('Frame absent from tracking')


def coordinates(group):
    try:
        x, y = map(float, group.split(',')[:2])
        return (x, y) if math.isfinite(x) and math.isfinite(y) else None
    except (ValueError, TypeError):
        return None


def inspect(frame, video, ffmpeg, output):
    import numpy as np
    meta, raw, homography = require(['atd_meta', 'atd_raw', 'homography'])
    with open(homography, newline='') as f:
        row = next((r for r in csv.DictReader(f) if int(r['frame']) == frame), None)
    if row is None or row['valid'] != '1':
        raise ValueError('No valid homography for this frame')
    matrix = np.array([float(row['h{}{}'.format(i, j)]) for i in range(3) for j in range(3)]).reshape(3, 3)
    players, ball = row_at(raw, frame)
    teams = team_of_each_track(meta)
    filters = []
    counts = {'H': 0, 'A': 0, 'ball': 0}
    for group, team in zip(players, teams):
        xy = coordinates(group)
        if xy is None:
            continue
        x, y = (float(v) for v in project(matrix, *xy))
        if math.isfinite(x) and math.isfinite(y) and 0 <= x < 1920 and 0 <= y < 1080:
            color = 'cyan' if team == 0 else 'yellow'
            counts['H' if team == 0 else 'A'] += 1
            filters.append('drawbox=x={}:y={}:w=18:h=18:color={}:t=3'.format(round(x)-9,round(y)-9,color))
    xy = coordinates(ball)
    if xy is not None:
        x, y = (float(v) for v in project(matrix, *xy))
        if math.isfinite(x) and math.isfinite(y) and 0 <= x < 1920 and 0 <= y < 1080:
            filters.append('drawbox=x={}:y={}:w=28:h=28:color=red:t=4'.format(round(x)-14,round(y)-14))
            counts['ball'] = 1
    command = [ffmpeg, '-v', 'error', '-ss', str(frame/GAMES[GAME]['fps']), '-i', video,
               '-map', '0:v:0', '-frames:v', '1']
    if filters:
        command += ['-vf', ','.join(filters)]
    command += ['-y', output]
    subprocess.run(command, check=True)
    print('Frame {}: home {} cyan, away {} yellow, ball {} red. Saved {}'.format(frame, counts['H'], counts['A'], counts['ball'], output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--video', required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not 0 <= args.frame < GAMES[GAME]['n_video_frames']:
        parser.error('frame outside tactical video')
    inspect(args.frame, args.video, args.ffmpeg, args.output)
