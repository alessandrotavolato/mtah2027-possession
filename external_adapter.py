"""Streaming DFL XML adapter, with possession/status labels isolated from inputs.

Source: Scientific Data article doi:10.1038/s41597-025-04505-y and figshare
10.6084/m9.figshare.28196177. Coordinates are centre-origin metres. BallStatus
0=inactive/1=active; BallPossession 1=home/2=away. Unspecified labels remain X.
Output per-half npz: distance, outside, features, feature_names, gt, frame_ids,
fps, ball_xy, pitch and team_ids. Features never include possession or status.
"""
import argparse
from datetime import datetime
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

FEATURE_NAMES = ['home_distance', 'away_distance', 'ball_speed', 'pitch_margin',
                 'player_count', 'mean_player_speed']


def frame_stream(path):
    """Yield Frame attributes and FrameSet context with bounded XML memory."""
    context = None
    for event, element in ET.iterparse(path, events=('start', 'end')):
        if event == 'start' and element.tag == 'FrameSet':
            context = dict(element.attrib)
        elif event == 'end' and element.tag == 'Frame':
            yield context, dict(element.attrib)
            element.clear()
        elif event == 'end' and element.tag == 'FrameSet':
            element.clear()
            context = None


def metadata(path):
    root = ET.parse(path).getroot()
    general, environment = root.find('.//General'), root.find('.//Environment')
    return (general.get('HomeTeamId'), general.get('GuestTeamId')), (
        float(environment.get('PitchX')), float(environment.get('PitchY')))


def parse_features(positions, match):
    """Read coordinate-only inputs. Label attributes are never inspected here."""
    teams, pitch = metadata(match)
    ball = {}
    for context, row in frame_stream(positions):
        if context is None or context.get('TeamId') != 'BALL':
            continue
        half = context['GameSection']
        ball.setdefault(half, []).append((int(row['N']), float(row['X']), float(row['Y']), row['T']))
    outputs = {}
    for half, records in ball.items():
        ids = np.array([r[0] for r in records], dtype=int)
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate ball frame IDs in '+half)
        first, last = ids.min(), ids.max()
        frame_ids = np.arange(first, last+1)
        xy = np.full((len(frame_ids), 2), np.nan)
        xy[ids-first] = np.array([[r[1],r[2]] for r in records])
        timed = sorted(records)
        periods = [(datetime.fromisoformat(b[3])-datetime.fromisoformat(a[3])).total_seconds()/(b[0]-a[0])
                   for a,b in zip(timed[:-1],timed[1:]) if b[0]>a[0]]
        fps = 1/np.median(periods)
        margin = np.minimum(pitch[0]/2-np.abs(xy[:,0]),pitch[1]/2-np.abs(xy[:,1]))
        speed = np.r_[np.nan, np.linalg.norm(np.diff(xy,axis=0),axis=1)*fps]
        outputs[half] = dict(frame_ids=frame_ids, ball_xy=xy, fps=float(fps),
            distance=np.full((len(frame_ids),2),np.inf), outside=margin < -0.5,
            speed=speed, margin=margin, count=np.zeros(len(frame_ids)),
            player_speed_sum=np.zeros(len(frame_ids)),player_speed_count=np.zeros(len(frame_ids)),
            pitch=np.array(pitch),team_ids=np.array(teams))
    previous = {}
    for context, row in frame_stream(positions):
        if context is None or context.get('TeamId') not in teams:
            continue
        half = context['GameSection']
        if half not in outputs:
            continue
        output=outputs[half]
        fid=int(row['N']);i=fid-output['frame_ids'][0]
        if not 0 <= i < len(output['frame_ids']):
            continue
        xy=np.array([float(row['X']),float(row['Y'])])
        if not np.isfinite(xy).all():
            continue
        team=teams.index(context['TeamId'])
        d=np.linalg.norm(xy-output['ball_xy'][i])
        if np.isfinite(d):
            output['distance'][i,team]=min(output['distance'][i,team],d)
        output['count'][i]+=1
        key=(half,context['PersonId'])
        if key in previous and fid==previous[key][0]+1:
            speed=np.linalg.norm(xy-previous[key][1])*output['fps']
            output['player_speed_sum'][i]+=speed
            output['player_speed_count'][i]+=1
        previous[key]=(fid,xy)
    for output in outputs.values():
        output['distance'][~np.isfinite(output['distance'])]=np.nan
        speed_sum=output.pop('player_speed_sum');speed_count=output.pop('player_speed_count')
        mean=np.divide(speed_sum,speed_count,out=np.full(len(output['frame_ids']),np.nan),where=speed_count>0)
        output['features']=np.column_stack([output['distance'],output.pop('speed'),output.pop('margin'),output.pop('count'),mean])
        output['feature_names']=np.array(FEATURE_NAMES)
    return outputs


def parse_labels(positions, feature_halves):
    """Independent label extraction; the output must never enter model inputs."""
    labels={half:np.full(len(v['frame_ids']),'X',dtype='<U1') for half,v in feature_halves.items()}
    for context,row in frame_stream(positions):
        if context is None or context.get('TeamId')!='BALL':continue
        half=context['GameSection']
        if half not in labels:continue
        i=int(row['N'])-feature_halves[half]['frame_ids'][0]
        status=row.get('BallStatus')
        if status=='0':labels[half][i]='D'
        elif status=='1':labels[half][i]={'1':'H','2':'A'}.get(row.get('BallPossession'),'X')
    return labels


def convert(positions,match,directory):
    halves=parse_features(positions,match)
    labels=parse_labels(positions,halves)
    Path(directory).mkdir(parents=True,exist_ok=True)
    for half,values in halves.items():
        path=Path(directory)/(half+'.npz')
        np.savez_compressed(path,**values,gt=labels[half])
        print(path,len(labels[half]),values['fps'],dict(zip(*np.unique(labels[half],return_counts=True))))
    combined={key:np.concatenate([value[key] for value in halves.values()]) for key in ('features','distance','outside','frame_ids','ball_xy')}
    combined['gt']=np.concatenate([labels[half] for half in halves])
    combined['half']=np.concatenate([np.full(len(value['frame_ids']),half) for half,value in halves.items()])
    combined['feature_names']=np.array(FEATURE_NAMES)
    combined['fps']=next(iter(halves.values()))['fps']
    np.savez_compressed(str(directory)+'.npz',**combined)
    return halves,labels


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--positions',required=True);p.add_argument('--match',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();convert(a.positions,a.match,a.output)
