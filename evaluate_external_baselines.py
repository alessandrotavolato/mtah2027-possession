"""Fixed baselines on a distinct external test match; no parameter selection."""
import json
import numpy as np
from pathlib import Path
from starter.baseline import smooth
from hybrid_solution import predict
from evaluation.possession_metrics import evaluate,label_codes

if __name__=='__main__':
    data=np.load('../external_validation/J03WQQ.npz')
    rows=[]
    for half in np.unique(data['half']):
        mask=data['half']==half;d=data['distance'][mask];o=data['outside'][mask]
        raw=np.where(np.isfinite(d).all(axis=1),np.argmin(np.nan_to_num(d,nan=1e6),axis=1),-1)
        m=tuple(np.zeros(len(d),bool) for _ in range(3))
        for name,labels in [('smooth50',smooth(raw,50)),('smooth125',smooth(raw,125)),('hybrid',predict(d,o,m,use_tags=False,fps=25))]:
            r=evaluate(label_codes(data['gt'][mask]),label_codes(labels),bootstrap_n=0)
            rows.append(dict(half=str(half),model=name,**{k:r[k] for k in ('macro_f1','spell_f1_mean','n_gt_spells','n_pred_spells')}))
    Path('../trained_models').mkdir(exist_ok=True)
    Path('../trained_models/external_baselines.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))
