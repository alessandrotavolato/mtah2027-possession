# Token Gobblers — ball possession by Viterbi decoding: technical document

More than a Hack 2027, Phase 1. Match BAR-ATH, tactical source.

## 1. Approach

The ranking metric, spellF1, scores possessions as whole objects. A per-frame classifier
followed by smoothing does not fit that metric well: a small window flickers and invents possessions, and
a large window removes the flicker but makes every change of possession late. We therefore treat the
match as a sequence and decode it in one pass:

1. **Per-frame evidence.** For every frame we compute a log-score for each of three states,
   H (home in possession), A (away in possession) and D (dead ball), from the tracking data and
   the Smart Tagging.
2. **Viterbi decoding.** A three-state Viterbi pass finds the single best label sequence for each
   half. Every change of state has a fixed cost: `cost_switch` for H↔A and `cost_dead` for
   entering or leaving D. A brief touch or deflection cannot accumulate enough evidence to pay for
   a switch, but sustained control can. Viterbi looks both forward and backward in time,
   so a switch lands on the frame where the evidence flips rather than half a window later.
3. **Snapping dead-ball edges to events.** Each D stretch is moved so that it ends on the
   restart kick and starts on the line crossing, whenever one is found within 3 s.

Everything is deterministic: no learning, no training data, and about one second of runtime after
the first parse.

## 2. Evidence used

| signal | source | role |
|---|---|---|
| distance from the nearest home player and the nearest away player to the ball | ATD players + ball | H vs A |
| ball speed | ATD ball, differentiated | gates the H/A evidence; detects kicks |
| ball outside the pitch lines (0.5 m margin) | ATD ball | D |
| "pending restart": from the ball leaving the pitch until the next kick or throw | ATD ball | D |
| median player speed (1 s rolling mean) | ATD players | D (players stand still in stoppages) |
| ball not tracked | ATD ball | weak D hint |
| set-piece tags (kick-offs, throw-ins, corners, goal kicks, free kicks) with team | Smart Tagging | weak D before the restart, weak team prior after it |

Details:

- **Team evidence** is `0.5 × clip(d_away − d_home, ±4 m)`, multiplied by two factors that fall to zero:
  - a *proximity* factor, which is 1 within 2 m of the ball and 0 beyond 5 m. Nobody controls a ball they are not near;
  - a *control* factor, which is 1 below 6 m/s ball speed and 0 above 12 m/s. A ball passing a player at speed is
    a pass or a shot going by, not possession.
- **Ball gaps** of up to 1 s are linearly interpolated. Longer gaps are left empty. 41% of frames
  have no ball, but almost all of that missing time comes in runs longer than 10 s. Most of those
  runs are stoppages after the ball has left the camera's view near a line. In the remaining runs
  the players are moving fast, and the player-speed signal tells the two cases apart.
- **Kick** means a tracked, in-bounds ball travelling faster than 7 m/s and at least 3 m/s faster than
  it was 5 frames earlier. A ball carried or rolled back to the restart spot is slower than that,
  so it does not end the stoppage.

## 3. Assets used and dropped

- **Used:** tactical ATD tracking (players and ball), ATD metadata (which team each track belongs to), and Smart Tagging
  (set pieces only).
- **Not used:** the homography. The tracking is already in pitch coordinates and we detect no objects in the image. Nor
  the broadcast video and tracking (on a different timeline, with no clear gain), nor the tactical
  video (used only for visual spot checks, never as input). Smart Tagging *phase* tags
  (build-up, progression, …) were not used: they are phases, not possessions, and their
  boundaries are too coarse for frame-level transitions.

## 4. Training data

None in the submitted model. It has no learned parameters. All 20-odd constants (see `PARAMS` in
`solution/model.py`) were set from football reasoning and physical units: player speeds, ball
speeds and distances. We used no hand labels on any game.

The public 5-minute ground truth was used only to check correctness: frame alignment, label
meaning, and what kinds of error occur. We did not use it to fit or choose parameter values. The one
global value chosen by reasoning about the output, `cost_switch = 16`, was set so that the
full-match mean possession length (13.7 s over 272 possessions) is in the typical range
for professional football. The challenge documentation quotes 14.2 s.

### Alternatives we tried and dropped

The team built three approaches in parallel and compared them before choosing this one.

- **Touch-sequence rule + supervised dead-ball classifier.** Possession changes only at a
  *controlled* contact (ball within 0.6 m of a player for at least 0.4 s, or two consecutive contacts
  by the same team); lone touches that the original team wins back are deflections. Ball in play vs
  dead came from a gradient-boosting classifier (scikit-learn `HistGradientBoostingClassifier`) on
  team-level features: tempo at several time scales, ball-visibility gaps, last and next ball
  position relative to the lines, and spread.
- **Tuning this model's constants on external games.** A leave-one-game-out random search over
  800 settings of the `PARAMS` above, on the external games with noise added to imitate the
  automatic tracking. Held-out spellF1 there rose from 0.446 to 0.485.
- **A two-stage model trained on DFL open data**, explored by one team member.

The external data used by these experiments is Metrica Sports' public sample data
(github.com/metrica-sports/sample-data), games 1 and 2, roughly 286,000 frames. Metrica's analysts
annotated the events. We turned those events into frame labels mechanically: dead from the event
before each set piece until the set piece; otherwise the team of the most recent on-ball event.
Nothing came from the provided game.

**Why we dropped them.** On the external games the classifier approach scored higher than this
model (spellF1 0.685 vs about 0.49). On the provided match it did not. Its dead-ball classifier found
only 40% of the dead frames in the public window, against 75% for this model. The tuned constants
also did worse on the public window than the hand-set ones. Our reading: the sample games' tracking
is cleaner and their labels are derived from events, so what is learned there does not carry
over to this match's automatic tracking, where the ball is missing 41% of the time.

**Use of the public window, stated plainly.** That choice between finished candidates used the
public 5 minutes, together with the external-game results. Each candidate was scored on it once.
No parameter of the submitted model was fitted or adjusted on it. We kept the hand-set
constants rather than tuning them further on 21 segments.

## 5. Possession rules implemented

- **Change of possession** requires the other team to be closest to a *slow* ball (one that can be controlled)
  for long enough to outweigh `cost_switch`. In practice that means roughly a second of clear
  control. Touches, deflections and balls flying past players do not change possession.
- **Loose and contested balls**, where nobody is close or the ball is fast, give no team evidence, so
  possession stays with the last team, as the rules require.
- **Dead ball** begins when the ball crosses a line, or when play visibly stops (players almost
  stationary). It lasts until the restart kick or throw. The restarting team, which is whoever is nearest
  the ball at the kick, takes possession from that frame.
- **Dead-ball ownership** is not modelled: D is always plain D, as the task specifies.

## 6. Results (public 5-minute window — a sanity check only, 17 possessions)

| submission | macroF1 | spellF1 |
|---|---|---|
| this model | 0.844 | 0.547 |
| baseline_smooth50 (published floor) | 0.621 | 0.326 |

Full match: 272 possessions, mean 13.7 s; frames are 35.4% H, 23.0% A and 41.6% D. The public window is
too small to rank models (EVALUATION.md), so these numbers show only that the pipeline is correct
and that the approach is not worse than the baseline.

## 7. Limitations and next steps

- **Stoppages without a line crossing** (fouls, offside, injuries) are caught only through player
  speed, which falls off gradually, so these D stretches start late or are missed entirely when
  they are short. A whistle detector based on the sudden deceleration of all players together would help.
- **Tracking noise near the ball.** Identity switches and missing players can give the ball to
  the wrong team for a moment. This is usually absorbed by the switch cost, but some switches are a
  few seconds early or late.
- **Long ball gaps during live play** carry no team evidence, so the decoder keeps the previous team
  until the ball reappears. A ball detector on the tactical video could fill these gaps.
- The constants are hand-set. With labelled data from other games they could be learned
  (for example as an HMM or CRF) instead. Our experiments show that the training games would need
  automatic tracking like this match's (with gaps, identity switches and a frequently missing ball)
  for that to help.

## 8. Reproduction

```bash
conda create -n mtah python=3.11 && conda activate mtah
pip install -r solution/requirements.txt
# put the data bundle in ./data (see DATA.md) or set MTAH_DATA_DIR
python solution/model.py "Token Gobblers.csv"
python evaluation/validate_submission.py "Token Gobblers.csv"
```

Tested with Python 3.11 in a fresh virtual environment containing only the two pinned packages.
The first run parses the 200 MB tracking file into `data/cache_tactical.npz` (a few seconds).
Output is deterministic: two runs produce byte-identical files.
