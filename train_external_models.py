"""Train on one external half, select on its other half, test on another match.
Only DFL other-match labels are read. Competition GT is never loaded.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from evaluation.possession_metrics import evaluate, label_codes


def expand_features(x, fps=25):
    # Label-free rolling summaries retain context without frame identifiers.
    parts=[x]
    for seconds in (1,3):
        roll=pd.DataFrame(x).rolling(int(fps*seconds),center=True,min_periods=1)
        parts.extend([roll.mean().values,roll.std().values])
    return np.concatenate(parts,axis=1)


def decode_probabilities(prob, penalty):
    if penalty==0:
        return np.argmax(prob,axis=1)
    score=np.log(np.clip(prob,1e-8,1))
    n=len(score);back=np.zeros((n,3),np.int8);value=score[0].copy()
    trans=np.full((3,3),-penalty);np.fill_diagonal(trans,0)
    for i in range(1,n):
        candidates=value[:,None]+trans
        back[i]=np.argmax(candidates,axis=0)
        value=candidates[back[i],np.arange(3)]+score[i]
    out=np.zeros(n,np.int8);out[-1]=np.argmax(value)
    for i in range(n-1,0,-1):out[i-1]=back[i,out[i]]
    return out


def score(y,p):
    r=evaluate(label_codes(y),label_codes(np.array(['H','A','D'])[p]),bootstrap_n=0)
    return {k:r[k] for k in ('macro_f1','spell_f1_mean','n_gt_spells','n_pred_spells')}


def main(train_path,test_path,out):
    train=np.load(train_path,allow_pickle=False);test=np.load(test_path,allow_pickle=False)
    assert np.array_equal(train['feature_names'],test['feature_names'])
    # Per-half rolling prevents half-boundary leakage in the feature expansion.
    def prepare(data):
        halves=data['half'];x=np.empty((len(halves),data['features'].shape[1]*5))
        for half in np.unique(halves):
            mask=halves==half;x[mask]=expand_features(data['features'][mask])
        return x,data['gt'].astype('<U1'),halves
    x,y,half=prepare(train);xt,yt,ht=prepare(test)
    halves=np.unique(half);train_mask=(half==halves[0])&(y!='X');dev_mask=half==halves[1]
    mapping={'H':0,'A':1,'D':2};yc=np.array([mapping.get(v,-1) for v in y])
    # Subsample only training examples, not validation/test predictions.
    ids=np.flatnonzero(train_mask)[::5]
    # Label-preserving augmentation masks ball-derived inputs in contiguous
    # runs; player motion/count remain available. This models occlusion only,
    # not the full camera/tracker failure process.
    def occlude(base, halves, seed):
        rng=np.random.default_rng(seed);z=base.copy()
        for h in np.unique(halves):
            ix=np.flatnonzero(halves==h);mask=np.zeros(len(ix),bool)
            for _ in range(max(1,len(ix)//300)):
                start=int(rng.integers(len(ix)));length=int(rng.integers(25,501))
                mask[start:start+length]=True
            z[np.ix_(ix[mask],np.arange(4))]=np.nan
        return z
    augmented=occlude(train['features'],half,37)
    xa=np.empty_like(x)
    for h in np.unique(half):
        mask=half==h;xa[mask]=expand_features(augmented[mask])
    fit_x=np.concatenate([x[ids],xa[ids]])
    fit_y=np.concatenate([yc[ids],yc[ids]])
    models={
       'logistic':make_pipeline(SimpleImputer(add_indicator=True),StandardScaler(),LogisticRegression(max_iter=500,C=1)),
       'random_forest':make_pipeline(SimpleImputer(add_indicator=True),RandomForestClassifier(n_estimators=100,max_depth=14,min_samples_leaf=30,n_jobs=2,random_state=17)),
       'boosted_trees':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=1,random_state=17,early_stopping=False)}
    records=[];trained={};best=None
    for name,model in models.items():
        model.fit(fit_x,fit_y);trained[name]=model
        prob=model.predict_proba(x[dev_mask])
        for penalty in (0,5,15,40):
            result=score(y[dev_mask],decode_probabilities(prob,penalty))
            row=dict(model=name,penalty=penalty,**result);records.append(row)
            print('development',row,flush=True)
            key=(result['spell_f1_mean'],result['macro_f1'])
            if best is None or key>best[0]:best=(key,name,penalty)
    _,name,penalty=best
    # Freeze selection before opening test labels for metric computation.
    tests=[]
    for h in np.unique(ht):
        mask=ht==h;pred=decode_probabilities(trained[name].predict_proba(xt[mask]),penalty)
        tests.append(dict(half=str(h),**score(yt[mask],pred)))
    stressed=occlude(test['features'],ht,911)
    stress_tests=[]
    for h in np.unique(ht):
        mask=ht==h
        pred=decode_probabilities(trained[name].predict_proba(expand_features(stressed[mask])),penalty)
        stress_tests.append(dict(half=str(h),**score(yt[mask],pred)))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=dict(train_match=Path(train_path).name,test_match=Path(test_path).name,
        selected=dict(model=name,penalty=penalty),development=records,test=tests,occlusion_stress_test=stress_tests,
        note='Other-match validation; DFL labels and tracking differ from the competition data. No competition GT used.')
    (out/'external_results.json').write_text(json.dumps(result,indent=2))
    joblib.dump(dict(model=trained[name],penalty=penalty,feature_names=train['feature_names'].tolist()),out/'selected_model.joblib')
    print('FINAL',json.dumps(result['selected']),json.dumps(tests),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--test',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.train,a.test,a.out)
