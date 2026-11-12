#!/usr/bin/env python3
"""Descriptive only, external SkillCorner matches, aggregates over the 20: how do SkillCorner's player-possession
events (``*_dynamic_events.csv``, in the pinned open-data commit, never used for labels) relate to the frame label
we train and score on (``possession.group``, null -> X)?

    KATENA_EXTERNAL=... uv run python scripts/probe_sc_actions.py

It reads the canonical SkillCorner matches and the events CSVs. It fits nothing, scores nothing and selects
nothing. The "action owner" of a frame is the team of the most recent player possession of the same period. An
"interruption gap" runs from the end of a player possession to the start of the next one when either carries
SkillCorner's ``game_interruption_after`` / ``game_interruption_before`` flag (throw-in, goal kick, corner, free
kick, goal).
"""

from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

ext = Path(os.environ["KATENA_EXTERNAL"])
H, A, X = 0, 1, 3


def gaps_and_spells() -> None:
    tot = np.zeros(12)
    agree_n = agree_d = 0
    spells = []
    for f in sorted(glob.glob(str(ext / "skillcorner/repo/data/matches/*/*_dynamic_events.csv"))):
        mid = Path(f).parent.name
        mj = json.loads((Path(f).parent / f"{mid}_match.json").read_text())
        home = str(mj["home_team"]["id"])
        z = np.load(ext / f"canonical/skillcorner/{mid}.npz", allow_pickle=True)
        y = z["y"]; n = len(y)
        bdet = z["extras__ball_detected"] if "extras__ball_detected" in z.files else None
        if bdet is None:
            cand = [k for k in z.files if "ball_det" in k]; bdet = z[cand[0]]
        pp = [r for r in csv.DictReader(open(f)) if r["event_type"] == "player_possession"]
        pp.sort(key=lambda r: int(r["frame_start"]))
        owner = np.full(n, -1, np.int8)      # team of the most recent player possession (carried forward within a period)
        inposs = np.zeros(n, bool)
        interrupt = np.zeros(n, bool)
        periods = [(p["period"], p["start_frame"], min(p["end_frame"], n - 1)) for p in mj["match_periods"]]
        for k, r in enumerate(pp):
            s, e = int(r["frame_start"]), min(int(r["frame_end"]), n - 1)
            t = H if r["team_id"] == home else A
            nxt = int(pp[k + 1]["frame_start"]) if k + 1 < len(pp) and pp[k + 1]["period"] == r["period"] else \
                [b for p, a, b in periods if str(p) == r["period"]][0] + 1
            owner[s:nxt] = t
            inposs[s:e + 1] = True
            if r["game_interruption_after"] or (k + 1 < len(pp) and pp[k + 1]["game_interruption_before"] and pp[k + 1]["period"] == r["period"]):
                interrupt[e + 1:nxt] = True
        inplay = np.zeros(n, bool)
        for _, a, b in periods: inplay[a:b + 1] = True
        lab = inplay & (y != X); unk = inplay & (y == X)
        agree_n += int((owner[lab] == y[lab]).sum()); agree_d += int(lab.sum())
        same_next = np.zeros(n, bool)       # gap frames whose next possession is by the same team as the last one
        for a_, b_ in zip(pp, pp[1:]):
            if a_["period"] == b_["period"] and a_["team_id"] == b_["team_id"]:
                same_next[int(a_["frame_end"]) + 1:int(b_["frame_start"])] = True
        gap = unk & ~inposs & ~interrupt & (owner >= 0)
        tot += [inplay.sum(), lab.sum(), unk.sum(), (unk & inposs).sum(), (unk & interrupt).sum(), (gap & same_next).sum(),
                (gap & ~same_next).sum(), (unk & (owner < 0)).sum(), (unk & (bdet < 0)).sum(), (unk & interrupt & (bdet < 0)).sum(),
                (lab & interrupt).sum(), (lab & ~inposs).sum()]
        # spells of the action-based label: owner, with interruptions as dead
        lbl = np.where(interrupt, 2, owner)
        for _, a, b in periods:
            seg = lbl[a:b + 1]; cut = np.flatnonzero(np.diff(seg) != 0) + 1
            st = np.r_[0, cut]; en = np.r_[cut, len(seg)]
            spells += [((e - s) / 10) for s, e in zip(st, en) if seg[s] in (0, 1)]
    names = ["inplay", "labelled H/A", "X", "X inside a player possession", "X in an interruption gap", "X between two possessions of one team",
             "X between possessions of different teams", "X before any possession", "X with no tracking", "X interruption with no tracking",
             "labelled inside an interruption gap", "labelled outside any player possession"]
    for k, v in zip(names, tot): print(f"{k:45s} {v / tot[0]:.3f} of in-play frames")
    print("team agreement of the action owner with the frame label, on labelled frames:", round(agree_n / agree_d, 4))
    sp = np.array(spells)
    print("action-based possessions per match:", round(len(sp) / 20), "| median s", round(float(np.median(sp)), 1), "| under 2 s", round(float((sp < 2).mean()), 3),
          "| under 4 s", round(float((sp < 4).mean()), 3), "| over 16 s", round(float((sp > 16).mean()), 3))


def disagreement() -> None:
    c = dict(lab=0, dis=0, dis_in=0, dis_out=0, dis_out_lead=0)
    runs = []; pp_inside_other = pp_n = 0; pp_inside_dur = []; start_types = {}
    lag = []
    for f in sorted(glob.glob(str(ext / "skillcorner/repo/data/matches/*/*_dynamic_events.csv"))):
        mid = Path(f).parent.name
        mj = json.loads((Path(f).parent / f"{mid}_match.json").read_text())
        home = str(mj["home_team"]["id"])
        y = np.load(ext / f"canonical/skillcorner/{mid}.npz", allow_pickle=True)["y"]; n = len(y)
        pp = [r for r in csv.DictReader(open(f)) if r["event_type"] == "player_possession"]
        pp.sort(key=lambda r: int(r["frame_start"]))
        owner = np.full(n, -1, np.int8); inposs = np.zeros(n, bool)
        pend = {str(p["period"]): min(p["end_frame"], n - 1) for p in mj["match_periods"]}
        for k, r in enumerate(pp):
            s, e = int(r["frame_start"]), min(int(r["frame_end"]), n - 1); t = H if r["team_id"] == home else A
            nxt = int(pp[k + 1]["frame_start"]) if k + 1 < len(pp) and pp[k + 1]["period"] == r["period"] else pend[r["period"]] + 1
            owner[s:nxt] = t; inposs[s:e + 1] = True
            seg = y[s:e + 1]; segl = seg[seg != X]
            if len(segl):
                pp_n += 1
                if (segl != t).mean() > 0.5:          # the frame label gives most of this player possession to the other team
                    pp_inside_other += 1; pp_inside_dur.append((e - s + 1) / 10)
                    start_types[r["start_type"]] = start_types.get(r["start_type"], 0) + 1
            # when the team changes at this possession: how many frames before its start did the frame label change?
            if k and pp[k - 1]["team_id"] != r["team_id"] and pp[k - 1]["period"] == r["period"] and not r["game_interruption_before"]:
                a0 = int(pp[k - 1]["frame_end"]); w = y[a0:s + 1]
                idx = np.flatnonzero(w == t)
                if len(idx) and y[min(s + 5, n - 1)] == t:
                    lag.append((s - (a0 + idx[0])) / 10)
        lab = (y != X) & (owner >= 0); dis = lab & (owner != y)
        c["lab"] += lab.sum(); c["dis"] += dis.sum(); c["dis_in"] += (dis & inposs).sum(); c["dis_out"] += (dis & ~inposs).sum()
        d = np.flatnonzero(np.diff(np.r_[0, dis.astype(np.int8), 0]))
        runs += list((d[1::2] - d[::2]) / 10)
    runs = np.array(runs)
    print("disagreement share of labelled frames:", round(c["dis"] / c["lab"], 3), "| inside a player possession:", round(c["dis_in"] / c["lab"], 3), "| in the gaps between:", round(c["dis_out"] / c["lab"], 3))
    print("disagreement runs per match:", round(len(runs) / 20), "| median s", np.median(runs), "| p90", np.quantile(runs, .9), "| share of disagreeing time in runs under 2 s:", round(runs[runs < 2].sum() / runs.sum(), 3), "| under 5 s:", round(runs[runs < 5].sum() / runs.sum(), 3))
    print("player possessions the frame label gives to the other team:", pp_inside_other, "of", pp_n, "=", round(pp_inside_other / pp_n, 3), "| per match", round(pp_inside_other / 20), "| median s", np.median(pp_inside_dur))
    tot = sum(start_types.values()); print("their start types:", sorted(((k, round(v / tot, 2)) for k, v in start_types.items()), key=lambda x: -x[1])[:6])
    lag = np.array(lag); print("at a change of team: the frame label changes this many s before the new team's first possession starts: median", np.median(lag), "| p25", np.quantile(lag, .25), "| p75", np.quantile(lag, .75), "| n per match", round(len(lag) / 20))


if __name__ == "__main__":
    gaps_and_spells()
    disagreement()
