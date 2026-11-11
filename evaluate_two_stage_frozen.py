"""Frozen-only external evaluation: no fitting, decoding selection or target GT."""
import argparse
import json
import numpy as np
import joblib
from sklearn.metrics import f1_score
from train_external_models import expand_features,decode_probabilities,score
from two_stage_models import control_features,probabilities,decode,occlude


def metrics(gt,prediction):
    result=score(gt,prediction)
    labels=np.array(['H','A','D'])[prediction]
    result['per_class_f1']=dict(zip(['H','A','D'],f1_score(gt,labels,labels=['H','A','D'],average=None,zero_division=0).tolist()))
    return result


def evaluate(new,old,path):
    data=np.load(path);result={}
    for condition,raw in [('clean',data['features']),('occlusion',occlude(data['features'],data['half'],911))]:
        result[condition]=[]
        for half in np.unique(data['half']):
            m=data['half']==half
            p=decode(probabilities(new,control_features(raw[m],float(data['fps']))),new['team_penalty'],new['dead_penalty'])
            prev=decode_probabilities(old['model'].predict_proba(expand_features(raw[m,:6],float(data['fps']))),old['penalty'])
            result[condition].append(dict(half=str(half),two_stage=metrics(data['gt'][m],p),frozen_previous=metrics(data['gt'][m],prev)))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--previous',required=True);p.add_argument('--tests',required=True,nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args();new=joblib.load(a.model);old=joblib.load(a.previous)
    result={path:evaluate(new,old,path) for path in a.tests}
    with open(a.output,'w') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
