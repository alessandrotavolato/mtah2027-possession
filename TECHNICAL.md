# Token Gobblers: ball possession by Viterbi decoding

Alessandro Tavolato, Francesco Tavolato, Carl Weber, Keo Chupin, Laslo Rose.

More than a Hack 2027, Phase 1, BAR-ATH (tactical source).

## 1. Approach

spellF1 rewards whole possessions, so we decode the match as a sequence instead of classifying frames and smoothing afterwards. Smoothing with a small window leaves flicker; a large one makes every change late.

1. For each frame we compute a score for three states: H, A and D (dead ball).
2. A Viterbi pass per half picks the best label sequence. Each change of state has a cost (`cost_switch` for H/A, `cost_dead` for entering or leaving D), so a single touch can't flip possession but a second or so of clear control can. Because Viterbi sees the whole half, switches land where the evidence actually changes, not half a window later.
3. Each dead-ball stretch is then snapped to physical events: it ends at the restart kick and starts at the line crossing, if either is found within 3 s.

The model is deterministic, has no learned parameters and runs in about a second once the tracking is parsed.

## 2. Evidence

| signal | source | used for |
|---|---|---|
| distance of nearest home and nearest away player to the ball | ATD | H vs A |
| ball speed | ATD | gating team evidence, detecting kicks |
| ball outside the lines (0.5 m margin) | ATD | D |
| "pending restart": ball has left the pitch and no kick yet | ATD | D |
| median player speed, 1 s rolling mean | ATD | D (players stand still at stoppages) |
| ball not tracked | ATD | weak D |
| set-piece tags with team | Smart Tagging | weak D before the restart, weak team prior after it |

Team evidence is `0.5 × clip(d_away − d_home, ±4 m)`. It fades to zero when nobody is within 5 m of the ball (full strength within 2 m) and when the ball moves faster than 6 to 12 m/s, since a ball flying past a player isn't under control.

Ball gaps up to 1 s are interpolated. The ball is missing on 41% of frames, mostly in long runs. Most of those are stoppages after the ball leaves the camera's view near a line; in the rest, players are running, and player speed separates the two cases.

A kick is an in-bounds ball above 7 m/s that sped up by at least 3 m/s over 5 frames. Carrying the ball back to the spot is slower, so it doesn't end a stoppage.

## 3. Assets used and dropped

Used: tactical ATD tracking (players and ball), ATD metadata (team of each track), Smart Tagging set-piece tags.

Not used: the homography (the tracking is already in pitch coordinates), the broadcast video and tracking (different timeline, no clear gain), the tactical video (only watched for spot checks, never an input), and the Smart Tagging phase tags (too coarse for frame-level changes).

## 4. Training data

The submitted model uses none. Its ~20 constants (`PARAMS` in `solution/model.py`) were set by hand from football and physical reasoning: distances, player and ball speeds. No hand labels on any game. `cost_switch` was set so the full-match mean possession comes out at a realistic length (13.7 s; the challenge docs quote 14.2 s).

Our dropped experiments (section 5) used Metrica Sports' public sample data, games 1 and 2 (github.com/metrica-sports/sample-data, about 286,000 frames). The events were annotated by Metrica. We turned them into frame labels with fixed rules: dead from the event before each set piece until the set piece is taken, otherwise the team of the last on-ball event. Nothing came from the provided game.

## 5. What we tried and dropped

We built three approaches in parallel:

- **Touch rule plus a dead-ball classifier.** Possession changes only on a controlled contact (ball within 0.6 m of a player for 0.4 s, or two contacts in a row by the same team); deflections won back by the original team don't count. In-play vs dead came from a gradient-boosting classifier trained on the Metrica games, using team tempo, ball-visibility gaps, last and next ball position relative to the lines, and player spread.
- **Tuning our constants on the Metrica games**, with noise added to look like automatic tracking. Leave-one-game-out search over 800 settings; held-out spellF1 went from 0.446 to 0.485.
- **A two-stage model trained on DFL open data.**

The classifier approach beat ours on the Metrica games (spellF1 0.685 vs about 0.49) but not on this match. In the public window it caught 40% of dead-ball frames against our 75%. The tuned constants also did worse there than the hand-set ones. Our explanation: the sample games have cleaner tracking and event-derived labels, so what is learned from them doesn't transfer to automatic tracking where the ball is missing this often.

To be explicit: we used the public 5 minutes to choose between these finished candidates, each scored once, alongside the external results. No parameter of the submitted model was fitted on it.

## 6. Possession rules

- Possession changes only when the other team is closest to a slow, controllable ball for about a second. Touches, deflections and balls passing by don't change it.
- Loose or contested balls (nobody close, or ball moving fast) carry no team evidence, so possession stays with the last team.
- Dead ball starts when the ball crosses a line or play visibly stops, and lasts until the restart kick or throw. Whoever is nearest at the kick gets possession from that frame.
- D is never assigned to a team.

## 7. Results on the public window (sanity check only)

| | macroF1 | spellF1 |
|---|---|---|
| our model | 0.844 | 0.547 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 |

Full match: 272 possessions, mean 13.7 s; 35.4% H, 23.0% A, 41.6% D. Five minutes is too little to rank models, so this mostly shows the pipeline is aligned and working.

## 8. Limitations

- Stoppages without a line crossing (fouls, offsides, injuries) are only picked up through player speed, which drops gradually, so these start late or are missed when short. Detecting the moment all players decelerate together would help.
- Identity switches and missing players near the ball can briefly give it to the wrong team. The switch cost absorbs most of this, but some changes end up a few seconds off.
- During long ball gaps in live play there is no team evidence, so the previous team is kept. A ball detector on the tactical video would fill these.
- The constants are hand-set. They could be learned (e.g. an HMM or CRF), but our experiments suggest the training games would need tracking as noisy as this match's for that to pay off.

## 9. Reproduction

```bash
conda create -n mtah python=3.11 && conda activate mtah
pip install -r solution/requirements.txt
# put the data bundle in ./data (see DATA.md) or set MTAH_DATA_DIR
python solution/model.py "Token Gobblers.csv"
python evaluation/validate_submission.py "Token Gobblers.csv"
```

Tested in a fresh environment with only the two pinned packages. The first run parses the tracking into `data/cache_tactical.npz`. Two runs give byte-identical output.
