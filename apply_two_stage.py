"""Apply frozen two-stage model to coordinate-only NPZ. No labels are read."""
import argparse
import numpy as np
import joblib
from two_stage_models import prepare,probabilities,decode
from starter.baseline import write_submission


def apply(model,features,output):
    bundle=joblib.load(model);data=np.load(features)
    if data['feature_names'].tolist()!=bundle['feature_names']:
        raise ValueError('Feature names/order differ from training')
    p=probabilities(bundle,prepare(data))
    labels=np.array(['H','A','D'])[decode(p,bundle['team_penalty'],bundle['dead_penalty'])]
    empty=(~np.isfinite(data['distance']).any(axis=1))&(data['features'][:,4]<2)
    labels[empty]='X';write_submission(output,labels)
    print(dict(zip(*np.unique(labels,return_counts=True))))
    return labels


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();apply(a.model,a.features,a.output)
