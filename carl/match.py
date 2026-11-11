"""A match as plain arrays, so the same pipeline runs on BAR-ATH and on external validation games.

    M = dict(P=(n, slots, 3) x/y normalised 0-1 + speed m/s,
             B=(n, 3) ball x/y normalised (+ unused),
             team=(slots,) 0 home / 1 away,
             blocks=[(half, start, stop_inclusive), ...]  graded spans,
             sign1=+1 if home attacks towards x=1 in the first half else -1,
             length=m, width=m, fps=25)
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def attack_sign_from_meta(meta_path):
    import xml.etree.ElementTree as ET
    root = ET.parse(meta_path).getroot()
    team = root.find(".//Teams/Team")
    for prm in team.iter("ProviderParameter"):
        if prm.findtext("Name") == "attack_direction_first_half":
            return 1.0 if prm.findtext("Value") == "left_to_right" else -1.0
    raise ValueError("attack direction not found")


def barca():
    from games import GAME, GAMES, half_video_blocks, require
    from parse_atd import load
    P, B, team = load()
    return dict(P=P, B=B, team=team, blocks=half_video_blocks(GAME),
                sign1=attack_sign_from_meta(require(["atd_meta"])[0]),
                length=GAMES[GAME]["pitch"][0], width=GAMES[GAME]["pitch"][1],
                fps=GAMES[GAME]["fps"])
