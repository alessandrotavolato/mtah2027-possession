"""Two-stage external-only learning and frozen evaluation.

Train DFL J03WOH first half, select on its second half (clean + synthetic
occlusion), freeze, and evaluate J03WQQ plus a fresh other match. No competition
labels read. Stage 1 predicts live/dead; stage 2 predicts home/away conditional
on live play. Decoder penalties are selected only on the development half.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from train_external_models import expand_features,decode_probabilities,score
from starter.baseline import write_submission


def occlude(base,halves,seed):
    rng=np.random.default_rng(seed);z=base.copy()
    for h in np.unique(halves):
        ix=np.flatnonzero(halves==h);mask=np.zeros(len(ix),bool)
        for _ in range(max(1,len(ix)//300)):
            start=int(rng.integers(len(ix)));length=int(rng.integers(25,501))
            mask[start:start+length]=True
        z[np.ix_(ix[mask],np.r_[np.arange(4),np.arange(6,z.shape[1])])]=np.nan
    return z


def control_features(x,fps=25):
    """Relative control + persistent motion; no status/possession annotations."""
    h,a=x[:,0],x[:,1]
    minimum=np.minimum(h,a)
    relative=(a-h)/(1+h+a)
    extra=np.column_stack([a-h,relative,minimum,np.maximum(h,a),
        np.tanh((a-h)/2),np.isfinite(h)&np.isfinite(a),x[:,2]<0.5])
    base=np.column_stack([x,extra])
    parts=[base]
    frame=pd.DataFrame(base)
    for seconds in (1,3):
        rolling=frame.rolling(round(fps*seconds),center=True,min_periods=1)
        parts.extend([rolling.mean().values,rolling.std().values])
    # Fraction of missing ball observations distinguishes no evidence from a
    # stable zero speed. A centered window is allowed for offline prediction.
    missing=(~np.isfinite(h)|~np.isfinite(a)).astype(float)
    parts.append(pd.Series(missing).rolling(round(3*fps),center=True,min_periods=1).mean().values[:,None])
    return np.concatenate(parts,axis=1)


def prepare(data,raw=None):
    base=data['features'] if raw is None else raw
    halves=data['half'] if 'half' in data else np.zeros(len(base),dtype=int)
    nfeatures=control_features(base[:min(len(base),10)]).shape[1]
    result=np.empty((len(base),nfeatures))
    for half in np.unique(halves):
        mask=halves==half;result[mask]=control_features(base[mask],float(data['fps']))
    return result


def model_factory(name):
    if name=='boosted':
        return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=29,early_stopping=False)
    return make_pipeline(SimpleImputer(add_indicator=True),RandomForestClassifier(n_estimators=100,max_depth=14,min_samples_leaf=30,n_jobs=2,random_state=29))


def probabilities(bundle,x):
    live=bundle['live'].predict_proba(x)[:,1]
    home=bundle['team'].predict_proba(x)[:,0]
    return np.column_stack([live*home,live*(1-home),1-live])


def decode(prob,team_penalty,dead_penalty):
    if team_penalty==dead_penalty:return decode_probabilities(prob,team_penalty)
    log=np.log(np.clip(prob,1e-8,1));n=len(prob)
    transition=np.array([[0,-team_penalty,-dead_penalty],[-team_penalty,0,-dead_penalty],[-dead_penalty,-dead_penalty,0.]])
    back=np.zeros((n,3),np.int8);value=log[0].copy()
    for i in range(1,n):
        candidates=value[:,None]+transition;back[i]=np.argmax(candidates,axis=0)
        value=candidates[back[i],np.arange(3)]+log[i]
    result=np.zeros(n,np.int8);result[-1]=np.argmax(value)
    for i in range(n-1,0,-1):result[i-1]=back[i,result[i]]
    return result


def evaluate_models(bundle,old,data):
    # No fitting or parameter selection occurs here.
    result={}
    for condition,raw in [('clean',data['features']),('occlusion',occlude(data['features'],data['half'],911))]:
        result[condition]=[]
        for half in np.unique(data['half']):
            mask=data['half']==half
            newx=control_features(raw[mask],float(data['fps']))
            oldx=expand_features(raw[mask,:6],float(data['fps']))
            pred=decode(probabilities(bundle,newx),bundle['team_penalty'],bundle['dead_penalty'])
            oldpred=decode_probabilities(old['model'].predict_proba(oldx),old['penalty'])
            result[condition].append(dict(half=str(half),two_stage=score(data['gt'][mask],pred),frozen_previous=score(data['gt'][mask],oldpred)))
    return result


def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    data=np.load(args.train);half=data['half'];y=data['gt'];halves=np.unique(half)
    x=prepare(data);xa=prepare(data,occlude(data['features'],half,37))
    trainids=np.flatnonzero((half==halves[0])&(y!='X'))[::5]
    fitx=np.concatenate([x[trainids],xa[trainids]])
    fity=np.tile(y[trainids],2)
    dev=half==halves[1];devx=x[dev];stressx=xa[dev];devy=y[dev]
    records=[];best=None;models={}
    for name in ('boosted','forest'):
        live=model_factory(name);team=model_factory(name)
        live.fit(fitx,(fity!='D').astype(int))
        alive=fity!='D';team.fit(fitx[alive],(fity[alive]=='A').astype(int))
        bundle=dict(live=live,team=team)
        probs=[probabilities(bundle,z) for z in (devx,stressx)]
        models[name]=bundle
        for team_penalty in (5,20,50):
            for dead_penalty in (5,20,50):
                metrics=[score(devy,decode(p,team_penalty,dead_penalty)) for p in probs]
                row=dict(model=name,team_penalty=team_penalty,dead_penalty=dead_penalty,clean=metrics[0],occlusion=metrics[1])
                records.append(row)
                key=(np.mean([r['spell_f1_mean'] for r in metrics]),np.mean([r['macro_f1'] for r in metrics]))
                if best is None or key>best[0]:best=(key,name,team_penalty,dead_penalty)
                print('DEVELOPMENT',json.dumps(row),flush=True)
    key,name,tp,dp=best;bundle=models[name]
    bundle.update(team_penalty=tp,dead_penalty=dp,feature_names=data['feature_names'].tolist(),kind='two_stage_control',selection='mean development clean+occlusion spellF1; macroF1 tiebreak')
    # Persist frozen selection before accessing any external test labels.
    joblib.dump(bundle,output/'two_stage_model.joblib')
    old=joblib.load(args.previous)
    result=dict(selected=dict(model=name,team_penalty=tp,dead_penalty=dp),development=records,tests={})
    if not args.skip_test:
        for path in args.tests:
            result['tests'][Path(path).stem]=evaluate_models(bundle,old,np.load(path))
    (output/'results.json').write_text(json.dumps(result,indent=2))
    target=np.load(args.target)
    if target['feature_names'].tolist()!=bundle['feature_names']:
        raise ValueError('Target feature names/order differ from frozen training artifact')
    features=prepare(target)
    labels=np.array(['H','A','D'])[decode(probabilities(bundle,features),tp,dp)]
    empty=(~np.isfinite(target['distance']).any(axis=1))&(target['features'][:,4]<2)
    labels[empty]='X'
    write_submission(str(output/'two_stage_prediction.csv'),labels)
    print('FINAL',json.dumps(result['selected']),json.dumps(result['tests']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',required=True);parser.add_argument('--previous',required=True)
    parser.add_argument('--tests',nargs='+',default=[]);parser.add_argument('--skip-test',action='store_true');parser.add_argument('--target',required=True);parser.add_argument('--output',required=True)
    train(parser.parse_args())
