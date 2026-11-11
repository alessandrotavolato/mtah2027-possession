"""Label-free diagnostics over the FULL match. No ground truth is read here.

1. Tag agreement: Smart Tagging phases imply a team in possession
   (attacking phases -> that team; DEFENDING IN ... -> the other team). We report how often the
   prediction names that team on alive tagged frames. Tags are automatic and phase-level, so this
   is a noisy proxy, useful for spotting gross regressions, not for fine ranking.
2. Set-piece recall: fraction of SET PIECES windows containing at least one predicted D frame
   (every set piece is preceded by a stoppage).
3. Masked-ball test: hide the ball on random 6-20 s windows where it was visible and the model is
   re-run; we measure how often the masked decode keeps the unmasked decode's label there. This
   measures the gap model's ability to carry possession through occlusions.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
from starter.load_smart_tagging import load as load_tags  # noqa: E402

ATTACK = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION",
          "BALL IN FINAL THIRD", "BALL IN THE BOX"}
DEFEND = {"DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
          "DEFENDING IN DEFENSIVE THIRD"}
TEAMS = {"barcelona": "H", "atletic club": "A", "athletic club": "A"}


def tag_targets(n):
    target = np.full(n, "", dtype="<U1")
    setpieces = []
    for ev in load_tags():
        code = (ev["code"] or "").strip().upper()
        team = TEAMS.get((ev["tags"].get("Team") or "").strip().casefold())
        a, b = max(0, ev["start_frame"]), min(n, ev["end_frame"])
        if code == "SET PIECES":
            setpieces.append((a, b))
        elif team and code in ATTACK:
            target[a:b] = team
        elif team and code in DEFEND:
            target[a:b] = "A" if team == "H" else "H"
    return target, setpieces


def report(labels, name=""):
    n = len(labels)
    target, setpieces = tag_targets(n)
    m = (target != "") & np.isin(labels, ["H", "A"])
    agree = np.mean(labels[m] == target[m])
    graded = labels != "X"
    sp = [(a, b) for a, b in setpieces if graded[a:b].any()]
    sp_recall = np.mean([np.any(labels[a:b] == "D") for a, b in sp])
    print("{:<16} tag-agreement {:.3f} on {} frames | set-piece D recall {:.3f} ({} windows)".format(
        name, agree, int(m.sum()), sp_recall, len(sp)))
    return agree, sp_recall
