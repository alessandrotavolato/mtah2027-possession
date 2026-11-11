"""Apply the externally selected frozen model; never read competition labels."""
import argparse
import joblib
import numpy as np
from train_external_models import expand_features,decode_probabilities
from starter.baseline import write_submission

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--features',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();bundle=joblib.load(a.model);data=np.load(a.features)
    assert list(data['feature_names'])==bundle['feature_names']
    probability=bundle['model'].predict_proba(expand_features(data['features']))
    labels=np.array(['H','A','D'])[decode_probabilities(probability,bundle['penalty'])]
    # With no ball or players, decline to extrapolate a classifier beyond inputs.
    no_evidence=(~np.isfinite(data['distance']).any(axis=1))&(data['features'][:,4]<2)
    labels[no_evidence]='X'
    write_submission(a.output,labels)
    print(dict(zip(*np.unique(labels,return_counts=True))))
