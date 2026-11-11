"""Self-supervised native-resolution video gap-recovery experiment.

No human labels. Observed ATD locations are withheld inside sampled gaps and
used only after prediction to measure tracking-proxy error. Match split by
alternating 5-minute video blocks. Set MTAH_DATA_DIR; video path is an argument.
Requires opencv-python-headless, numpy and pandas. Never publish match images.
"""
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from games import GAME, GAMES, require
from starter.load_homography import load

FPS=25
PITCH=np.array(GAMES[GAME]['pitch'],dtype=float)
HALF_PATCH=7
SEARCH=35
SAMPLES_PER_SPLIT=80
SEED=4127
FAILURES={}

def failed(reason):
    FAILURES[reason]=FAILURES.get(reason,0)+1
    return None

def project(H,xy):
    z=H@np.r_[xy,1.0]
    if abs(z[2]) < 1e-10: return np.array([np.nan,np.nan])
    return z[:2]/z[2]

def load_ball():
    raw=require(['atd_raw'],GAME)[0]
    out=np.full((GAMES[GAME]['n_video_frames'],2),np.nan)
    with open(raw) as f:
        for line in f:
            a=line.strip().split(':')
            if len(a)!=3: continue
            i=int(a[0])-1
            b=a[2].split(',')
            if len(b)>=2 and b[0]!='NaN' and b[1]!='NaN':
                out[i]=float(b[0]),float(b[1])
    return out

def select_gaps(ball,width):
    seen=np.isfinite(ball).all(axis=1)
    missing=np.r_[0,np.cumsum(~seen)]
    candidates=np.arange(1,len(ball)-width-1)
    candidates=candidates[(missing[candidates+width+1]-missing[candidates-1])==0]
    rng=np.random.default_rng(SEED+width)
    rng.shuffle(candidates)
    bins=[[],[]]
    for x in candidates:
        block=x//(FPS*300)
        if (x+width)//(FPS*300)!=block: continue
        group=block%2
        if len(bins[group])>=SAMPLES_PER_SPLIT: continue
        if any(abs(int(x)-int(old)) < FPS*15 for subset in bins for old in subset): continue
        bins[group].append(int(x))
        if min(map(len,bins))>=SAMPLES_PER_SPLIT: break
    return bins

def extract(cap,start,end):
    if not cap.set(cv2.CAP_PROP_POS_FRAMES,start): return None
    frames=[]
    for _ in range(end-start+1):
        ok,img=cap.read()
        if not ok: return None
        frames.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
    return frames

def patch(gray,pt):
    if not np.isfinite(pt).all(): return None
    x,y=np.rint(pt).astype(int); h,w=gray.shape
    if x<HALF_PATCH or y<HALF_PATCH or x>=w-HALF_PATCH or y>=h-HALF_PATCH: return None
    p=gray[y-HALF_PATCH:y+HALF_PATCH+1,x-HALF_PATCH:x+HALF_PATCH+1]
    return p if np.std(p)>5 else None

def one_direction(frames,Hs,initial):
    n=len(frames); pos=np.full((n,2),np.nan); score=np.full(n,np.nan)
    pos[0]=initial
    pixel=project(Hs[0],initial)
    templ=patch(frames[0],pixel)
    if templ is None: return pos,score
    score[0]=1
    for i in range(1,n):
        center=project(Hs[i],pos[i-1]); gray=frames[i]
        if not np.isfinite(center).all(): break
        x,y=np.rint(center).astype(int); h,w=gray.shape
        x0=max(0,x-SEARCH-HALF_PATCH); x1=min(w,x+SEARCH+HALF_PATCH+1)
        y0=max(0,y-SEARCH-HALF_PATCH); y1=min(h,y+SEARCH+HALF_PATCH+1)
        search=gray[y0:y1,x0:x1]
        if search.shape[0]<templ.shape[0] or search.shape[1]<templ.shape[1]: break
        match=cv2.matchTemplate(search,templ,cv2.TM_CCOEFF_NORMED)
        _,s,_,loc=cv2.minMaxLoc(match)
        found=np.array([x0+loc[0]+HALF_PATCH,y0+loc[1]+HALF_PATCH],float)
        try: xy=project(np.linalg.inv(Hs[i]),found)
        except np.linalg.LinAlgError: break
        if not np.isfinite(xy).all() or np.any(xy<-.2) or np.any(xy>1.2): break
        pos[i]=xy; score[i]=s
    return pos,score

def estimate(cap,H,ball,start,width,with_image=False):
    # Gap is start..start+width-1; endpoints are just outside it.
    a=start-1; b=start+width
    frames=extract(cap,a,b)
    if frames is None: return failed("video decode or seek")
    mats=H[a:b+1]
    f,fs=one_direction(frames,mats,ball[a])
    br,bs=one_direction(frames[::-1],mats[::-1],ball[b]); back=br[::-1]; bs=bs[::-1]
    if not np.isfinite(f).all() or not np.isfinite(back).all():
        if not np.isfinite(f[1]).all() or not np.isfinite(back[-2]).all():
            return failed("endpoint patch unavailable or first tracking step failed")
        return failed("tracking lost after first step")
    agreement=np.linalg.norm((f-back)*PITCH,axis=1)
    t=np.linspace(0,1,len(frames))[:,None]
    recovered=(1-t)*f+t*back
    interp=(1-t)*ball[a]+t*ball[b]
    result={'agreement_m':float(np.median(agreement[1:-1])),
            'min_corr':float(min(np.nanmedian(fs),np.nanmedian(bs))),
            'start':start,'width':width,'recovered':recovered[1:-1], 'interp':interp[1:-1]}
    if with_image: result['frames']=frames
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--video',required=True);p.add_argument('--out',required=True); args=p.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    ball=load_ball(); H,err,valid=load(); cap=cv2.VideoCapture(args.video)
    if not cap.isOpened(): raise RuntimeError('Could not decode video')
    print('video frames:',int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),flush=True)
    report={'method':'native 15px fixed grayscale patch, bidirectional, homography-compensated, 35px search; linear trajectory blend',
      'split':'alternating nonoverlapping 5-minute video blocks', 'note':'Expanded exploratory sample overlaps earlier 24-window pilot; it is not fresh independent validation.', 'seed':SEED,'samples_per_split':SAMPLES_PER_SPLIT,
      'acceptance':'agreement threshold is calibration 25th percentile per duration; no tuning using held-out', 'durations':{}}
    examples=[]
    for seconds in [1,3,6]:
        width=seconds*FPS; groups=select_gaps(ball,width); rec=[[],[]]
        failures_by_split=[]
        for split in (0,1):
            FAILURES.clear()
            for k,start in enumerate(groups[split]):
                item=estimate(cap,H,ball,start,width)
                if item is not None: rec[split].append(item)
                if (k+1)%8==0: print(f'{seconds}s split {split}: {k+1}/{len(groups[split])}',flush=True)
            failures_by_split.append(dict(FAILURES))
        if len(rec[0])<10:
            report['durations'][str(seconds)]={'gate':'FAIL','reason':'fewer than 10 valid calibration windows','cal_candidates':len(groups[0]),'held_candidates':len(groups[1]),'cal_valid':len(rec[0]),'held_valid':len(rec[1]),'held_accepted':0,'coverage':0.0,'failures_by_split':failures_by_split}
            (out/'video_recovery_report.json').write_text(json.dumps(report,indent=2)+'\n')
            continue
        threshold=float(np.quantile([x['agreement_m'] for x in rec[0]],.25))
        accepted=[x for x in rec[1] if x['agreement_m']<=threshold]
        print('accepted',seconds,len(accepted),'threshold',threshold,flush=True)
        d={'cal_candidates':len(groups[0]),'cal_valid':len(rec[0]),'held_candidates':len(groups[1]),'held_valid':len(rec[1]),'held_accepted':len(accepted),'agreement_threshold_m':threshold,
           'coverage':len(accepted)/len(groups[1]) if groups[1] else 0, 'failures_by_split':failures_by_split}
        if accepted:
            errors=[];base=[]
            for x in accepted:
                truth=ball[x['start']:x['start']+width]
                errors.extend(np.linalg.norm((x['recovered']-truth)*PITCH,axis=1))
                base.extend(np.linalg.norm((x['interp']-truth)*PITCH,axis=1))
            d['video_error_median_m']=float(np.median(errors));d['video_error_p90_m']=float(np.percentile(errors,90))
            d['interpolation_error_median_m']=float(np.median(base));d['interpolation_error_p90_m']=float(np.percentile(base,90))
            d['gate']='PASS' if len(accepted)>=10 and d['video_error_median_m']<d['interpolation_error_median_m'] and d['video_error_p90_m']<d['interpolation_error_p90_m'] else 'FAIL'
            examples.append(accepted[0])
        else: d['gate']='FAIL';d['reason']='no accepted held-out gaps'
        report['durations'][str(seconds)]=d
        (out/'video_recovery_report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'video_recovery_report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Diagnostic image only: truth is revealed after prediction and evaluation.
    if examples:
        ex=examples[0]; i=ex['width']//2; frame=ex['start']+i
        imgs=extract(cap,frame,frame)
        if imgs:
            image=cv2.cvtColor(imgs[0],cv2.COLOR_GRAY2BGR)
            for label,xy,color in [('original tracking',ball[frame],(255,255,255)),('interpolation',ex['interp'][i],(0,255,255)),('video recovery',ex['recovered'][i],(255,100,0))]:
                pp=project(H[frame],xy)
                if np.isfinite(pp).all():
                    q=tuple(np.rint(pp).astype(int));cv2.circle(image,q,13,color,3);cv2.putText(image,label,(q[0]+15,q[1]),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
            cv2.imwrite(str(out/'video_diagnostic_local.png'),image)
    cap.release();print(json.dumps(report['durations'],indent=2),flush=True)

if __name__=='__main__':main()
