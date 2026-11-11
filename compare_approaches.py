"""Compare gap-recovery approaches using automated nearest-team proxy labels.
No possession ground truth is loaded. Development and test masks use distinct
seeds, but the same match: this is not independent match-level validation.
"""
import json
from pathlib import Path
import numpy as np
from temporal_solution import features
from benchmark_models import mask_windows
from starter.baseline import smooth, write_submission


def predict(distance,masks,method):
    seen=np.isfinite(distance).all(axis=1)
    indices=np.flatnonzero(seen)
    n=len(distance)
    if not len(indices):
        return np.full(n,'X')
    raw=np.full(n,-1,dtype=int);raw[seen]=np.argmin(distance[seen],axis=1)
    if method=='baseline':
        return smooth(raw,50)
    positions=np.arange(n)
    previous=np.clip(np.searchsorted(indices,positions,side='right')-1,0,len(indices)-1)
    following=np.clip(np.searchsorted(indices,positions,side='left'),0,len(indices)-1)
    left,right=indices[previous],indices[following]
    if method=='carry_forward':
        out=raw[left]
    elif method=='nearest_time':
        pick=np.where(positions-left<=right-positions,left,right)
        out=raw[pick]
    elif method in ('interpolate_distance','interpolate_tags'):
        d=np.column_stack([np.interp(positions,indices,distance[indices,t]) for t in (0,1)])
        if method=='interpolate_tags':
            home,away,restart=masks
            # Small additive prior in distance units; fixed rather than tuned.
            d[:,0]-=2*(home & ~away & ~restart)
            d[:,1]-=2*(away & ~home & ~restart)
        out=np.argmin(d,axis=1)
    else:
        raise ValueError(method)
    labels=np.array(['H','A'])[out]
    # Do not extrapolate across long natural gaps without evidence. All methods
    # abstain beyond 30 seconds from any observation; no missing=>dead rule.
    labels[(~seen)&(np.minimum(np.abs(positions-left),np.abs(right-positions))>750)]='X'
    return labels


if __name__=='__main__':
    distance,outside,masks=features()
    seen=np.isfinite(distance).all(axis=1)
    target=np.argmin(np.nan_to_num(distance,nan=1e6),axis=1)
    methods=['baseline','carry_forward','nearest_time','interpolate_distance','interpolate_tags']
    rows=[]
    for split,seeds in [('development',(7,19)),('test',(101,202,303))]:
        for sec in (1,3,6):
            for seed in seeds:
                held,count=mask_windows(seen & ~outside,sec*25,seed)
                masked=distance.copy();masked[held]=np.nan
                for method in methods:
                    p=predict(masked,masks,method)
                    correct=int(np.count_nonzero(p[held]==np.array(['H','A'])[target[held]]))
                    rows.append(dict(split=split,seconds=sec,seed=seed,method=method,correct=correct,total=int(held.sum())))
    summary={}
    for split in ('development','test'):
        summary[split]={}
        for method in methods:
            summary[split][method]={}
            for seconds in (1,3,6):
                rs=[r for r in rows if r['split']==split and r['method']==method and r['seconds']==seconds]
                summary[split][method][str(seconds)]=sum(r['correct'] for r in rs)/sum(r['total'] for r in rs)
    # Equal weight to each gap length, selected ONLY from development masks.
    winner=max(methods,key=lambda m:np.mean(list(summary['development'][m].values())))
    print(json.dumps(dict(selected_on_development=winner,scores=summary),indent=2))
    Path('approach_comparison.json').write_text(json.dumps(dict(selected_on_development=winner,scores=summary,rows=rows),indent=2))
    # All outputs remain experiments; no public score used for selection.
    for method in methods[1:]:
        write_submission('evaluation/results/'+method+'.csv',predict(distance,masks,method))
