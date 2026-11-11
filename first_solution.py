"""First experiment: bridge brief tracking gaps when both sides agree.

No match labels are used. Five seconds is a provisional, explainable assumption,
not a value fitted to the public sample. This is an experiment, not a verified
improvement. Run with MTAH_DATA_DIR set to the provided data directory.
"""
from pathlib import Path
import numpy as np
from starter.baseline import build, smooth, write_submission, NOBODY
from games import GAME, GAMES

MAX_GAP_SECONDS = 5
SMOOTHING_HALF_WINDOW = 50
OUTPUT = Path(__file__).resolve().parent / 'evaluation' / 'results' / 'first_solution.csv'


def bridge_gaps(codes, max_gap):
    """Fill short missing runs only if the teams on both ends are the same."""
    result = codes.copy()
    missing = codes == NOBODY
    edges = np.diff(np.r_[False, missing, False].astype(np.int8))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    filled_runs = filled_frames = 0
    for start, end in zip(starts, ends):
        if start > 0 and end < len(codes) and end - start <= max_gap and codes[start - 1] == codes[end]:
            result[start:end] = codes[start - 1]
            filled_runs += 1
            filled_frames += end - start
    return result, filled_runs, filled_frames


if __name__ == '__main__':
    raw = build()
    bridged, runs, frames = bridge_gaps(raw, int(MAX_GAP_SECONDS * GAMES[GAME]['fps']))
    labels = smooth(bridged, SMOOTHING_HALF_WINDOW)
    write_submission(str(OUTPUT), labels)
    print('Bridged {} gaps ({} frames); saved {}'.format(runs, frames, OUTPUT))
