"""IDSSE / DFL open data (Bassek et al. 2025, Sci Data 12:195, CC BY 4.0) -> `Match`.

Seven Bundesliga matches, 25 fps optical tracking with per-frame ball flags:

- `BallStatus`     1 alive, 0 dead
- `BallPossession` 1 home, 2 away. Present on dead frames too (who restarts / touched last).

Labels: y = D where BallStatus == 0, else H/A from BallPossession. Home/away comes from the
metadata `Team Role="home"|"guest"`, never from file order (the guest team is listed first in
these files). Frames outside the two halves do not exist in the source; we insert `HALF_GAP` X
rows between halves.

Why not `kloppy.sportec.load_open_tracking_data`: kloppy 3.18.0 builds a nested dict of every
frame of every object before constructing frames (several GB for one match), and by default
(`only_alive=True`) it drops every dead-ball frame, which are exactly the D labels we need. We
use kloppy's URL map (file ids copied below from `kloppy._providers.sportec.get_IDSSE_url`) and
stream the positions XML straight into numpy arrays instead.

Coordinates: DFL is centred metres with y pointing up the screen. We flip y (y -> -y) so the
handedness matches EPTS / the competition ATD after `katena.match.from_normalised` (y down).
Speeds: DFL `S` is km/h; we store m/s.
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
from lxml import etree

from ..match import A, D, FPS, H, Match
from .common import HALF_GAP, concat_halves, detect_attack_dir

# match_id -> (home, away, {kind: figshare file id}); ids from kloppy 3.18.0 get_IDSSE_url
IDSSE_MATCHES = {
    "J03WMX": ("1. FC Koeln", "FC Bayern Muenchen",
               {"meta": 51643475, "event": 51643493, "tracking": 51643514}),
    "J03WN1": ("VfL Bochum 1848", "Bayer 04 Leverkusen",
               {"meta": 51643472, "event": 51643496, "tracking": 51643517}),
    "J03WPY": ("Fortuna Duesseldorf", "1. FC Nuernberg",
               {"meta": 51643487, "event": 51643505, "tracking": 51643526}),
    "J03WOH": ("Fortuna Duesseldorf", "SSV Jahn Regensburg",
               {"meta": 51643478, "event": 51643499, "tracking": 51643520}),
    "J03WQQ": ("Fortuna Duesseldorf", "FC St. Pauli",
               {"meta": 51643484, "event": 51643508, "tracking": 51643529}),
    "J03WOY": ("Fortuna Duesseldorf", "F.C. Hansa Rostock",
               {"meta": 51643481, "event": 51643502, "tracking": 51643523}),
    "J03WR9": ("Fortuna Duesseldorf", "1. FC Kaiserslautern",
               {"meta": 51643490, "event": 51643511, "tracking": 51643532}),
}
# springernature.figshare.com (kloppy's host) sits behind a bot challenge; this is figshare's
# standard download host for the same file ids.
FIGSHARE_URL = "https://ndownloader.figshare.com/files/{file_id}"

GAME_SECTIONS = {"firstHalf": 1, "secondHalf": 2, "firstHalfExtra": 3, "secondHalfExtra": 4}
KMH_TO_MS = 1.0 / 3.6


def idsse_paths(match_id: str, root: str | os.PathLike) -> dict:
    folder = Path(root) / "raw" / "idsse" / match_id
    return {"meta": folder / f"{match_id}_meta.xml",
            "event": folder / f"{match_id}_events.xml",
            "tracking": folder / f"{match_id}_tracking.xml"}


def download_idsse(match_id: str, root: str | os.PathLike, kinds=("meta", "event", "tracking"),
                   log=print) -> dict:
    """Fetch the raw files for one match unless already present. Returns the paths."""
    paths = idsse_paths(match_id, root)
    for kind in kinds:
        dst = paths[kind]
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = FIGSHARE_URL.format(file_id=IDSSE_MATCHES[match_id][2][kind])
        log(f"  downloading {match_id} {kind} ...")
        tmp = dst.with_name(dst.name + ".part")
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dst)
    return paths


# -- metadata ----------------------------------------------------------------------------------

def read_dfl_meta(path) -> dict:
    """Teams (by Role), players, pitch size and result from a DFL matchinformation XML."""
    root = etree.parse(str(path)).getroot()
    info = root.find("MatchInformation")
    general = info.find("General").attrib
    env = info.find("Environment").attrib
    teams, players = {}, {}
    for t in info.find("Teams").iter("Team"):
        role = t.get("Role")
        if role not in ("home", "guest"):
            raise ValueError(f"unknown team role {role!r}")
        code = 0 if role == "home" else 1
        teams[t.get("TeamId")] = {"code": code, "name": t.get("TeamName"), "role": role}
        for p in t.iter("Player"):
            players[p.get("PersonId")] = {"team_id": t.get("TeamId"), "team": code,
                                          "shirt": int(p.get("ShirtNumber")),
                                          "position": p.get("PlayingPosition"),
                                          "starting": p.get("Starting") == "true"}
    if sorted(v["code"] for v in teams.values()) != [0, 1]:
        raise ValueError("expected exactly one home and one guest team")
    return {"match_id": general.get("MatchId"), "title": general.get("MatchTitle"),
            "result": general.get("Result"), "kickoff_time": general.get("KickoffTime"),
            "pitch": (float(env["PitchX"]), float(env["PitchY"])),
            "teams": teams, "players": players}


def read_dfl_kickoffs(events_path) -> dict:
    """{half: TeamLeft id} from the KickOff events, if the event file is available.

    The team listed as TeamLeft starts on the -x side, so it attacks +x in that half. Used only
    as a cross-check of the position-based attack direction.
    """
    out = {}
    for _, el in etree.iterparse(str(events_path), events=("end",), tag="KickOff"):
        half = GAME_SECTIONS.get(el.get("GameSection"))
        if half is not None and el.get("TeamLeft") and half not in out:
            out[half] = el.get("TeamLeft")
        el.clear()
    return out


# -- positions ---------------------------------------------------------------------------------

def parse_dfl_positions(path) -> dict:
    """Stream a DFL positions XML into numpy arrays, one FrameSet at a time.

    Returns {"ball": {half: dict(N, X, Y, Z, S, status, poss)},
             "people": {person_id: {"team_id": str, half: dict(N, X, Y, S)}}}.
    Peak memory is one FrameSet of Python floats (about 70k frames), not the whole file.
    """
    ball, people = {}, {}
    cols = None
    cur = None
    ctx = etree.iterparse(str(path), events=("start", "end"), tag=("FrameSet", "Frame"),
                          huge_tree=True)
    for event, el in ctx:
        if el.tag == "Frame":
            if event == "end":
                if cur is not None:
                    a = el.attrib
                    cols["N"].append(int(a["N"]))
                    cols["X"].append(float(a["X"]))
                    cols["Y"].append(float(a["Y"]))
                    cols["S"].append(float(a.get("S", "nan")))
                    if cur["is_ball"]:
                        if not cols["T0"]:
                            cols["T0"].append(a.get("T"))
                        cols["Z"].append(float(a.get("Z", "nan")))
                        cols["status"].append(int(a.get("BallStatus", "-1")))
                        cols["poss"].append(int(a.get("BallPossession", "-1")))
                el.clear()
                while el.getprevious() is not None:
                    del el.getparent()[0]
            continue
        # FrameSet
        if event == "start":
            half = GAME_SECTIONS.get(el.get("GameSection"))
            team_id = el.get("TeamId")
            if half is None:
                cur = None
                continue
            is_ball = team_id == "BALL"
            cur = {"half": half, "team_id": team_id, "pid": el.get("PersonId"),
                   "is_ball": is_ball}
            cols = {"N": [], "X": [], "Y": [], "S": []}
            if is_ball:
                cols.update(Z=[], status=[], poss=[], T0=[])
        else:
            if cur is not None:
                arr = {"N": np.asarray(cols["N"], dtype=np.int64),
                       "X": np.asarray(cols["X"], dtype=np.float32),
                       "Y": np.asarray(cols["Y"], dtype=np.float32),
                       "S": np.asarray(cols["S"], dtype=np.float32)}
                if cur["is_ball"]:
                    arr.update(Z=np.asarray(cols["Z"], dtype=np.float32),
                               status=np.asarray(cols["status"], dtype=np.int8),
                               poss=np.asarray(cols["poss"], dtype=np.int8),
                               T0=cols["T0"][0] if cols["T0"] else None)
                    if cur["half"] in ball:
                        raise ValueError(f"two ball FrameSets for half {cur['half']}")
                    ball[cur["half"]] = arr
                else:
                    rec = people.setdefault(cur["pid"], {"team_id": cur["team_id"]})
                    if cur["half"] in rec:
                        raise ValueError(f"two FrameSets for {cur['pid']} half {cur['half']}")
                    rec[cur["half"]] = arr
            cur, cols = None, None
            el.clear()
            while el.getprevious() is not None:
                del el.getparent()[0]
    return {"ball": ball, "people": people}


def labels_from_dfl(status: np.ndarray, poss: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-frame y from BallStatus / BallPossession. Unknown owners on alive frames are filled
    from the previous known owner in the half (counted in the returned diagnostics)."""
    y = np.full(len(status), D, dtype=np.int8)
    alive = status == 1
    owner = np.where(poss == 1, H, np.where(poss == 2, A, -1)).astype(np.int8)
    unknown = alive & (owner < 0)
    filled = owner.copy()
    if unknown.any():
        idx = np.where(owner >= 0, np.arange(len(owner)), -1)
        idx = np.maximum.accumulate(idx)
        first_known = int(np.flatnonzero(owner >= 0)[0]) if (owner >= 0).any() else -1
        idx[idx < 0] = first_known
        filled = owner[idx] if first_known >= 0 else np.full_like(owner, H)
    y[alive] = filled[alive]
    bad_status = int(np.sum(~np.isin(status, (0, 1))))
    return y, {"alive_owner_unknown": int(unknown.sum()), "status_unknown": bad_status}


def load_idsse(meta_path, tracking_path, events_path=None, gap: int = HALF_GAP,
               name: str | None = None) -> Match:
    meta = read_dfl_meta(meta_path)
    pos = parse_dfl_positions(tracking_path)
    teams, players = meta["teams"], meta["players"]
    halves = sorted(pos["ball"])
    if not halves:
        raise ValueError("no ball FrameSet found")

    # slot order: home then away, then by first appearance, then id (deterministic)
    def first_seen(pid):
        rec = pos["people"][pid]
        h = min(k for k in rec if isinstance(k, int))
        return h, int(rec[h]["N"][0])

    pids = [p for p, rec in pos["people"].items() if rec["team_id"] in teams]
    dropped_people = sorted(set(pos["people"]) - set(pids))  # officials etc., if present
    pids.sort(key=lambda p: (teams[pos["people"][p]["team_id"]]["code"], *first_seen(p), p))
    team = np.array([teams[pos["people"][p]["team_id"]]["code"] for p in pids], dtype=np.int8)
    S = len(pids)

    parts, diag = [], {"dropped_objects": dropped_people, "halves": {}}
    for half in halves:
        b = pos["ball"][half]
        n0, n1 = int(b["N"].min()), int(b["N"].max())
        n = n1 - n0 + 1
        rows = b["N"] - n0
        P = np.full((n, S, 2), np.nan, dtype=np.float32)
        speed = np.full((n, S), np.nan, dtype=np.float32)
        ball = np.full((n, 2), np.nan, dtype=np.float32)
        ball_z = np.full(n, np.nan, dtype=np.float32)
        status = np.full(n, -1, dtype=np.int8)
        poss = np.full(n, -1, dtype=np.int8)
        ball[rows, 0], ball[rows, 1] = b["X"], -b["Y"]
        ball_z[rows] = b["Z"]
        status[rows], poss[rows] = b["status"], b["poss"]
        outside = 0
        for s, pid in enumerate(pids):
            rec = pos["people"][pid].get(half)
            if rec is None:
                continue
            r = rec["N"] - n0
            keep = (r >= 0) & (r < n)
            outside += int((~keep).sum())
            P[r[keep], s, 0] = rec["X"][keep]
            P[r[keep], s, 1] = -rec["Y"][keep]
            speed[r[keep], s] = rec["S"][keep] * KMH_TO_MS
        y, ydiag = labels_from_dfl(status, poss)
        owner_raw = np.where(poss == 1, 0, np.where(poss == 2, 1, -1)).astype(np.int8)
        frame_id = np.arange(n0, n1 + 1, dtype=np.int64)
        parts.append(dict(half=half, P=P, speed=speed, ball=ball, y=y, ball_z=ball_z,
                          owner_raw=owner_raw, ball_status=status, frame_id=frame_id))
        diag["halves"][half] = {"first_N": n0, "last_N": n1, "n_frames": n,
                                "first_frame_time": b.get("T0"),
                                "ball_frames": int(len(rows)),
                                "player_frames_outside_ball_range": outside, **ydiag}

    arr, half_blocks = concat_halves(parts, gap=gap)
    attack_dir, ad_diag = detect_attack_dir(arr["P"], team, half_blocks, FPS)
    provider_dir = {}
    if events_path is not None and Path(events_path).exists():
        home_id = next(t for t, v in teams.items() if v["code"] == 0)
        for half, left in read_dfl_kickoffs(events_path).items():
            provider_dir[half] = 1 if left == home_id else -1
    for half, d in provider_dir.items():
        if half in attack_dir:
            ad_diag[half]["provider"] = d
            ad_diag[half]["provider_agrees"] = bool(d == attack_dir[half])

    home_name = next(v["name"] for v in teams.values() if v["code"] == 0)
    away_name = next(v["name"] for v in teams.values() if v["code"] == 1)
    m = Match(P=arr["P"], team=team, speed=arr["speed"], ball=arr["ball"],
              half_blocks=half_blocks, attack_dir=attack_dir, y=arr["y"],
              name=name or f"idsse-{meta['match_id'].split('-')[-1]}", fps=FPS,
              pitch=meta["pitch"],
              meta={"source": "idsse", "match_id": meta["match_id"], "title": meta["title"],
                    "result": meta["result"], "home": home_name, "away": away_name,
                    "slot_ids": pids, "coords": "centred metres, y flipped from DFL (y down)",
                    "speed_source": "provider S (km/h) / 3.6", "label_source":
                    "BallStatus/BallPossession", "attack_dir_diag": ad_diag, "diag": diag,
                    "frame_id": arr["frame_id"], "ball_z": arr["ball_z"],
                    "owner_raw": arr["owner_raw"], "ball_status": arr["ball_status"]})
    m.validate()
    return m
