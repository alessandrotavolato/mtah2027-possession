"""Coordinate-only competition features matching external_adapter.FEATURE_NAMES.

Player slots can swap identities: derived player speed is noisy. No possession
labels, BallStatus fields, public answer keys or event annotations are read.
"""
import numpy as np
from games import GAME, GAMES, require
from starter.baseline import team_of_each_track
from external_adapter import FEATURE_NAMES


def read_recoveries(path, frame_count):
    """Read real-gap positions that passed the video experiment's acceptance rule."""
    import csv
    recovered = {}
    if path:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                i = int(row["frame"])
                xy = np.array([float(row["x_m"]), float(row["y_m"])])
                confidence = float(row["confidence"])
                if not 0 <= i < frame_count or i in recovered:
                    raise ValueError(f"Duplicate or invalid recovery frame {i}")
                if not np.isfinite(xy).all() or not 0 <= confidence <= 1:
                    raise ValueError(f"Invalid recovery at frame {i}")
                recovered[i] = (xy, confidence)
    return recovered


def target_features(recovery_csv=None):
    meta,raw=require(['atd_meta','atd_raw'])
    teams=team_of_each_track(meta)
    config=GAMES[GAME];pitch=np.array(config['pitch']);fps=config['fps'];n=config['n_video_frames']
    distance=np.full((n,2),np.nan);ball_xy=np.full((n,2),np.nan)
    count=np.zeros(n);mean_player_speed=np.full(n,np.nan)
    recovered = read_recoveries(recovery_csv, n)
    source = np.full(n, "missing", dtype="<U8")
    # The supplied tracking has no confidence field. Preserve that distinction.
    confidence = np.full(n, np.nan)
    previous=None;previous_id=None
    with open(raw) as f:
        for line in f:
            fid,players,ball=line.rstrip().split(':');i=int(fid)-1
            p=np.array([[float(v) for v in g.split(',')[:2]] for g in players.split(';')])*pitch
            b=np.array([float(v) for v in ball.split(',')[:2]])*pitch
            if np.isfinite(b).all():
                if i in recovered:
                    raise ValueError(f"Recovery would replace observed ball at frame {i}")
                source[i] = "tracking"
            elif i in recovered:
                b, confidence[i] = recovered[i]
                source[i] = "video"
            ball_xy[i]=b
            valid=np.isfinite(p).all(axis=1);count[i]=valid.sum()
            if previous is not None and i==previous_id+1:
                speed=np.linalg.norm(p-previous,axis=1)*fps
                finite=speed[np.isfinite(speed)]
                if len(finite):mean_player_speed[i]=finite.mean()
            previous=p;previous_id=i
            if np.isfinite(b).all():
                distances=np.linalg.norm(p-b,axis=1)
                for team in (0,1):
                    d=distances[(teams==team)&np.isfinite(distances)]
                    if len(d):distance[i,team]=d.min()
    # Verify each inserted position belongs to a short, bounded ORIGINAL gap.
    if recovered:
        original_gap = source != "tracking"
        edges = np.diff(np.r_[False, original_gap, False].astype(np.int8))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if (source[start:end] == "video").any() and (
                    start == 0 or end == n or end - start > 6 * fps):
                raise ValueError(f"Recovery in unbounded or overlong gap {start}:{end}")
    speed=np.r_[np.nan,np.linalg.norm(np.diff(ball_xy,axis=0),axis=1)*fps]
    margin=np.minimum(ball_xy[:,0],pitch[0]-ball_xy[:,0])
    margin=np.minimum(margin,np.minimum(ball_xy[:,1],pitch[1]-ball_xy[:,1]))
    features=np.column_stack([distance,speed,margin,count,mean_player_speed])
    return {'features':features,'feature_names':np.array(FEATURE_NAMES),'distance':distance,
            'outside':margin < -0.5,'frame_ids':np.arange(n),'fps':float(fps),'ball_xy':ball_xy, 'ball_source':source, 'video_confidence':confidence}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--recoveries')
    a=p.parse_args();values=target_features(a.recoveries);np.savez_compressed(a.output,**values)
    print(a.output,values['features'].shape,values['feature_names'].tolist())
