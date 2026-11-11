# Possession model (Carl) — touch rule + dead-ball classifier + Viterbi

Draft of the technical write-up for this approach. Numbers below are from the **external
validation games** unless marked "public window" (the 5 public minutes, used only as a pipeline
check — never to choose between versions).

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r carl/requirements.txt
# BAR-ATH tactical bundle in data/tactical/ (or MTAH_DATA_DIR)
git clone --depth 1 https://github.com/metrica-sports/sample-data ~/code/external/sample-data
.venv/bin/python carl/run.py          # -> evaluation/results/carl_possession.csv  (~25 s)
.venv/bin/python evaluation/validate_submission.py evaluation/results/carl_possession.csv
```

`EXTERNAL_DIR` overrides the sample-data location. The run is deterministic (fixed seeds); two
runs produce byte-identical files.

## Pipeline

1. **Parse** the tactical ATD into arrays (`parse_atd.py`, cached in `data/`, gitignored).
2. **Features** (`features.py`): ball interpolated over gaps ≤ 1 s; nearest home / away player
   distance to the ball (metres); ball speed; mean player speed ("tempo"); ball distance outside
   the pitch.
3. **Dead vs alive** (`dead_model.py` + `model.py`): a gradient-boosted classifier trained **only
   on Metrica's public sample games 1–2** (other matches, human event data — allowed by the
   rules). Features are team-level: tempo at 0.5/1/3/6 s and its change, max/90th-pct player
   speed, player spread, ball-seen fractions, time since/until the ball was seen, where it was
   last/next seen relative to touch and goal lines, ball speed, nearest-player distance. Its
   per-frame log-odds plus a hard "ball observed > 0.75 m outside the pitch → dead" rule are
   decoded with a 3-state Viterbi (switch penalties), giving clean dead-ball intervals.
4. **Who has it** (`touches.py`), inside each alive interval:
   - a *contact* is a run of frames with the ball within 0.8 m of a player, owned by the nearest
     player's team;
   - the first contact after a restart sets the possessor;
   - possession changes to the other team only on a **controlled** contact: held ≥ 0.4 s, or
     followed by a teammate's contact within 3 s (completed reception/pass), or a brief touch that
     **changes the ball's course** (direction > 35° or speed by > 50%) — an interception;
   - a ball passing near a player without changing course, or no contact at all (ball in flight
     or untracked), keeps the previous team — "loose balls stay with the last team".

## Validation (external games 1–2, official metric code)

Leave-one-game-out for the dead-ball classifier; also run on a degraded copy (7% of
player-frames dropped in multi-second segments, ball jittered 5 cm) to mimic automatic ATD.

| step | spellF1 clean | spellF1 degraded |
|---|---|---|
| nearest-player + Viterbi (≈ smoothed baseline idea) | 0.405 | — |
| + touch/control rule | 0.574 | 0.582 |
| + external dead-ball classifier | 0.61 | — |
| + course-change touch detection, 0.8 m radius (**final**) | 0.67 | 0.647 |

Differences below ~0.02 between settings were treated as noise and not chased.

Public window (pipeline check only): spellF1 0.522, macroF1 0.708, 19 possessions vs 17 true,
mean 14.1 s vs 14.2 s. Published baseline floor on the full match: 0.2433.

## Assets used / dropped

- **Used:** tactical ATD (players + ball), ATD metadata (team of each slot).
- **Dropped:** a self-supervised team-shape classifier for ball-missing stretches (lowered
  external spellF1 0.50 → 0.41: most missing-ball stretches are stoppages, so it pushed them to
  H/A); Smart Tagging (only used as a label-free sanity diagnostic, `diagnostics.py` — it rewards
  long phases, so it cannot rank segmentations); homography and video (not needed: ATD is already
  in pitch coordinates); broadcast feed.

## Training data

Metrica Sports public sample data, games 1 and 2 (github.com/metrica-sports/sample-data):
anonymised tracking + human-annotated events by Metrica. Labels derived mechanically
(`external.py`): dead from the end of the event preceding each set piece until the set piece;
owner = team of the latest on-ball event (pass, recovery, set piece, shot, ball lost, ball out).
No data from the competition match was labelled or used as a target.

## Limitations / next steps

- Short possessions (< 2 s, ~25% of spells in the sample games) are still mostly missed.
- Stoppages with the ball tracked *on* the pitch (fouls) are under-detected: in the sample
  games the ball is almost never tracked during stoppages, so the classifier leans on ball
  visibility. More external games (sample game 3, EPTS format) would help.
- Our transitions come ~0.4 s before the public truth's (lat_f ≈ −10 frames), consistent with
  the corrected tracking crediting possession slightly after first contact. Not corrected, since
  the only evidence for it is the public window.
