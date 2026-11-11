"""Fixed-seed artificial tracking dropout benchmark; no possession GT loaded.

Observed nearest-team labels are noisy proxies. Artificial gaps in well-observed
periods are not representative of natural missingness or stoppages. Results
cannot establish improved possession accuracy, spellF1, or choose a final entry.
"""
import csv
from pathlib import Path
import numpy as np
from temporal_solution import features, decode
from starter.baseline import smooth
from first_solution import bridge_gaps


def mask_windows(seen, width, seed):
    rng = np.random.default_rng(seed)
    bad = np.r_[0,np.cumsum(~seen)]
    starts = np.arange(1,len(seen)-width)
    starts = starts[(bad[starts+width+1]-bad[starts-1])==0]
    rng.shuffle(starts)
    held = np.zeros(len(seen),bool)
    count=0
    for start in starts:
        if not held[max(0,start-125):min(len(seen),start+width+125)].any():
            held[start:start+width]=True
            count+=1
        if count==100:
            break
    return held,count


if __name__=='__main__':
    distance,outside,masks=features()
    seen=np.isfinite(distance).all(axis=1)
    raw=np.where(seen,np.argmin(np.nan_to_num(distance,nan=1e6),axis=1),-1)
    rows=[]
    for seconds in (1,3,6):
        for seed in (7,19,41):
            held,count=mask_windows(seen & ~outside,seconds*25,seed)
            d=distance.copy();d[held]=np.nan
            o=outside.copy();o[held]=False
            masked=raw.copy();masked[held]=-1
            bridged,_,_=bridge_gaps(masked,125)
            predictions={'baseline':smooth(masked,50),'short_bridge':smooth(bridged,50),
                'temporal_tracking':decode(d,o,masks,False),'temporal_tags':decode(d,o,masks,True)}
            target=np.array(['H','A'])[raw[held]]
            for name,p in predictions.items():
                rows.append(dict(seconds=seconds,seed=seed,windows=count,frames=int(held.sum()),model=name,
                    agreement=float(np.mean(p[held]==target)),unknown=float(np.mean(p[held]=='X')),
                    dead=float(np.mean(p[held]=='D'))))
        for name in predictions:
            selected=[r for r in rows if r['seconds']==seconds and r['model']==name]
            print(seconds,name,'agreement',round(np.mean([r['agreement'] for r in selected])*100,2),'unknown',round(np.mean([r['unknown'] for r in selected])*100,2),flush=True)
    path=Path(__file__).resolve().parent/'benchmark_results.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
