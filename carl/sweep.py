"""Label-free sweep: score variants on the Smart Tagging proxy (no ground truth read).
Writes each variant to evaluation/results/sw_<name>.csv for later pipeline checks."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
from features import compute
import model
from diagnostics import report
from run import stats
from starter.baseline import write_submission

F = compute()
for name, v in json.loads(sys.argv[1]).items():
    lab = model.predict(v, F)
    print(name, json.dumps(v)); stats(lab); report(lab, "")
    write_submission(os.path.join(ROOT, "evaluation", "results", "sw_" + name + ".csv"), lab)
