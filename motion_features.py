"""Append nearest-player/ball co-motion measured from adjacent coordinates.

Input labels are copied unchanged as dataset metadata and never inspected to
calculate features. Relative velocities are in m/s, not differences in speed.
DFL player IDs persist; competition track slots may switch identities. Adjacent
slot continuity plus a 15m/s player gate rejects large jumps but cannot reject
plausible identity switches. Ball velocities above60m/s are withheld as suspect.
Missing positions/velocities remain NaN; no gap or half-boundary interpolation.
"""
import argparse
from pathlib import Path
import numpy as np
from external_adapter import frame_stream, metadata
from games import require,GAME,GAMES
from starter.baseline import team_of_each_track

NAMES=['home_relative_velocity','away_relative_velocity','home_nearest_speed','away_nearest_speed']
MAX_PLAYER_SPEED=15.
MAX_BALL_SPEED=60.


def velocity(current,previous,fps,limit):
    v=(current-previous)*fps
    good=np.isfinite(v).all(axis=-1)&(np.linalg.norm(v,axis=-1)<=limit)
    return np.where(np.expand_dims(good,-1),v,np.nan)


def append(base,extra,output):
    values={key:base[key] for key in base.files}
    values['features']=np.column_stack([base['features'],extra])
    values['feature_names']=np.r_[base['feature_names'],np.array(NAMES)]
    np.savez_compressed(output,**values)
    print(output,extra.shape,'finite fractions',np.round(np.isfinite(extra).mean(axis=0),3),flush=True)


def dfl(base,positions,match):
    teams,_=metadata(match);fps=float(base['fps']);n=len(base['features'])
    extra=np.full((n,4),np.nan);nearest=np.full((n,2),np.inf)
    halves=base['half'];ids=base['frame_ids'];ball=base['ball_xy']
    ballv=np.full_like(ball,np.nan);index={}
    for half in np.unique(halves):
        ix=np.flatnonzero(halves==half)
        valid=ids[ix[1:]]==ids[ix[:-1]]+1
        v=velocity(ball[ix[1:]],ball[ix[:-1]],fps,MAX_BALL_SPEED)
        ballv[ix[1:][valid]]=v[valid]
        index[half]=(int(ids[ix[0]]),int(ix[0]),len(ix))
    previous={}
    for context,row in frame_stream(positions):
        if context is None or context.get('TeamId') not in teams:continue
        half=context['GameSection']
        if half not in index:continue
        fid=int(row['N']);first,offset,length=index[half];local=fid-first
        if not 0<=local<length:continue
        i=offset+local;xy=np.array([float(row['X']),float(row['Y'])]);t=teams.index(context['TeamId'])
        key=(half,context['TeamId'],context['PersonId']);pv=np.array([np.nan,np.nan])
        if key in previous and previous[key][0]+1==fid:
            pv=velocity(xy,previous[key][1],fps,MAX_PLAYER_SPEED)
        previous[key]=(fid,xy)
        distance=np.linalg.norm(xy-ball[i])
        if np.isfinite(distance) and distance<nearest[i,t]:
            nearest[i,t]=distance
            extra[i,t]=np.linalg.norm(pv-ballv[i])
            extra[i,2+t]=np.linalg.norm(pv)
    return extra


def target(base):
    meta,raw=require(['atd_meta','atd_raw']);teams=team_of_each_track(meta)
    pitch=np.array(GAMES[GAME]['pitch']);fps=float(base['fps']);ball=base['ball_xy']
    n=len(ball);extra=np.full((n,4),np.nan);previous=None;previous_i=None
    ballv=np.full_like(ball,np.nan)
    ballv[1:]=velocity(ball[1:],ball[:-1],fps,MAX_BALL_SPEED)
    with open(raw) as f:
        for line in f:
            fid,players,_=line.rstrip().split(':');i=int(fid)-1
            p=np.array([[float(v) for v in group.split(',')[:2]] for group in players.split(';')])*pitch
            pv=np.full_like(p,np.nan)
            if previous is not None and previous_i+1==i:
                pv=velocity(p,previous,fps,MAX_PLAYER_SPEED)
            previous=p;previous_i=i
            distances=np.linalg.norm(p-ball[i],axis=1)
            for t in (0,1):
                slots=np.flatnonzero((teams==t)&np.isfinite(distances))
                if len(slots):
                    slot=slots[np.argmin(distances[slots])]
                    extra[i,t]=np.linalg.norm(pv[slot]-ballv[i])
                    extra[i,2+t]=np.linalg.norm(pv[slot])
    return extra


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--positions');p.add_argument('--match');p.add_argument('--target',action='store_true')
    a=p.parse_args();base=np.load(a.base)
    extra=target(base) if a.target else dfl(base,a.positions,a.match)
    append(base,extra,a.output)
