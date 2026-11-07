"""The FIFA/EPTS layout of the competition ATD, reported as booleans and counts only.

``katena.target.atd`` reads the raw file the way ``starter/baseline.py`` does: player groups in the
order of the ``<Player>`` elements, three fields per group (x, y, speed), the first ``<Team>`` as
home. Reproducing the published baseline files (``scripts/check_target.py``) cannot test those
assumptions, because the organisers' baseline makes the same ones. This module follows the format's
own chain instead (DataFormatSpecification -> PlayerChannelRef -> PlayerChannel -> Player) and says
whether they hold. Nothing it returns is a match value: no position, name, id or frame number.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


def _tag(el) -> str:
    return el.tag.rsplit("}", 1)[-1]


def _children(el, name: str) -> list:
    return [c for c in el if _tag(c) == name]


def _iter(root, name: str):
    return (el for el in root.iter() if _tag(el) == name)


def layout_report(meta_path: str | Path, n_frames: int | None = None) -> dict:
    """Booleans and counts describing the metadata layout (see the module docstring)."""
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in _iter(root, "Team")]
    players = [(p.get("id"), p.get("teamId")) for p in _iter(root, "Player")]
    player_ids = [p for p, _ in players]
    channel = {pc.get("id"): (pc.get("playerId"), pc.get("channelId")) for pc in _iter(root, "PlayerChannel")}
    score = next(_iter(root, "Score"), None)
    specs = list(_iter(root, "DataFormatSpecification"))

    order_ok, one_player, unknown_refs = [], [], 0
    group_counts, channel_sets, ball_sets, frames = Counter(), Counter(), Counter(), []
    for dfs in specs:
        frames.append((int(dfs.get("startFrame", -1)), int(dfs.get("endFrame", -1))))
        registers = _children(dfs, "SplitRegister")
        groups = [g for r in registers for g in _children(r, "SplitRegister")
                  if _children(g, "PlayerChannelRef")]
        group_counts[len(groups)] += 1
        order = []
        for g in groups:
            refs = [r.get("playerChannelId") for r in _children(g, "PlayerChannelRef")]
            unknown_refs += sum(r not in channel for r in refs)
            owners = {channel[r][0] for r in refs if r in channel}
            one_player.append(len(owners) == 1)
            order.append(owners.pop() if len(owners) == 1 else None)
            channel_sets[tuple(channel[r][1] for r in refs if r in channel)] += 1
        order_ok.append(order == player_ids[:len(order)] and len(order) == len(player_ids))
        for r in registers:
            balls = _children(r, "BallChannelRef")
            if balls:
                ball_sets[tuple(b.get("channelId") for b in balls)] += 1

    frames.sort()
    contiguous = all(b + 1 == c for (_, b), (c, _) in zip(frames, frames[1:]))
    return {
        "n_teams": len(teams),
        "n_players": len(players),
        "players_per_team": sorted(Counter(t for _, t in players).values()),
        "every_player_has_a_listed_team": all(t in teams for _, t in players),
        "first_team_is_local_team": (score.get("idLocalTeam") == teams[0]) if score is not None and teams else None,
        "n_format_specs": len(specs),
        "specs_contiguous": contiguous,
        "specs_cover_1_to_n": (frames[0][0] == 1 and frames[-1][1] == n_frames) if frames and n_frames else None,
        "player_groups_per_spec": dict(group_counts),
        "every_group_is_one_player": all(one_player),
        "group_order_matches_player_elements": all(order_ok) if order_ok else None,
        "channels_per_group": {",".join(k): v for k, v in channel_sets.items()},
        "ball_channels": {",".join(k): v for k, v in ball_sets.items()},
        "unknown_channel_refs": unknown_refs,
    }


def assumptions_hold(rep: dict) -> bool:
    """True when katena.target.atd (and the starter baseline) read this layout correctly: two
    teams, one player per group, groups in <Player> order in every spec, the same number of groups
    in every spec, three channels per group starting with x and y, and no dangling references."""
    chans = [c.split(",") for c in rep["channels_per_group"]]
    return bool(rep["n_teams"] == 2 and rep["every_player_has_a_listed_team"]
                and rep["n_format_specs"] >= 1 and len(rep["player_groups_per_spec"]) == 1
                and rep["every_group_is_one_player"] and rep["group_order_matches_player_elements"]
                and chans and all(len(c) == 3 and c[:2] == ["x", "y"] for c in chans)
                and rep["unknown_channel_refs"] == 0 and rep["first_team_is_local_team"] is not False)
