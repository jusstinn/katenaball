"""Tiny provider-format files written from `katena.synth.synthetic_match`, for offline tests.

Each writer emits the source format the real loader expects (DFL positions/matchinformation/
events XML, FIFA/EPTS metadata XML + raw TXT), so a loader test is a round trip: synthetic Match
-> provider files -> loader -> Match, compared with the original.
"""
from __future__ import annotations

import numpy as np

from katena.match import A, D, H
from katena.synth import synthetic_match


def tiny_match(seed: int = 3, half_minutes: float = 0.1):
    return synthetic_match(seed=seed, half_minutes=half_minutes, pre=0, gap=0, post=0)


# -- DFL ---------------------------------------------------------------------------------------

DFL_HOME, DFL_AWAY = "DFL-CLU-HOME01", "DFL-CLU-AWAY01"
DFL_STARTS = {1: 10000, 2: 100000}
DFL_SECTIONS = {1: "firstHalf", 2: "secondHalf"}


def dfl_person(s: int) -> str:
    return f"DFL-OBJ-{s:06d}"


def write_dfl(m, folder):
    """matchinformation, positions and events XML for `m` (guest team listed first)."""
    folder.mkdir(parents=True, exist_ok=True)
    team_ids = [DFL_HOME if t == 0 else DFL_AWAY for t in m.team]

    def players(tid):
        return "\n".join(
            f'<Player PersonId="{dfl_person(s)}" ShirtNumber="{s + 1}" FirstName="F" '
            f'LastName="L{s}" Shortname="P{s}" Starting="true" PlayingPosition="ZO" '
            f'TeamLeader="false" />' for s in range(m.n_slots) if team_ids[s] == tid)

    meta = f"""<?xml version="1.0" encoding="UTF-8"?>
<PutDataRequest>
  <MatchInformation>
    <General MatchId="DFL-MAT-TEST01" MatchTitle="Home:Away" Result="1:0" MatchDay="1"
             KickoffTime="2022-10-15T11:01:28.300+00:00" />
    <Environment PitchX="{m.pitch[0]:.2f}" PitchY="{m.pitch[1]:.2f}" />
    <Teams>
      <Team TeamId="{DFL_AWAY}" TeamName="Away FC" Role="guest"><Players>{players(DFL_AWAY)}</Players></Team>
      <Team TeamId="{DFL_HOME}" TeamName="Home FC" Role="home"><Players>{players(DFL_HOME)}</Players></Team>
    </Teams>
  </MatchInformation>
</PutDataRequest>
"""
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<PutDataRequest>", "<Positions>"]
    for half, first, last in m.half_blocks:
        n0 = DFL_STARTS[half]
        sec = DFL_SECTIONS[half]
        for s in range(m.n_slots):
            lines.append(f'<FrameSet GameSection="{sec}" MatchId="DFL-MAT-TEST01" '
                         f'TeamId="{team_ids[s]}" PersonId="{dfl_person(s)}">')
            for i, f in enumerate(range(first, last + 1)):
                x, y = m.P[f, s]
                sp = m.speed[f, s] * 3.6
                lines.append(f'<Frame N="{n0 + i}" T="t" X="{x:.2f}" Y="{-y:.2f}" D="0.00" '
                             f'S="{sp:.2f}" A="0.00" M="1"/>')
            lines.append("</FrameSet>")
        lines.append(f'<FrameSet GameSection="{sec}" MatchId="DFL-MAT-TEST01" TeamId="BALL" '
                     f'PersonId="DFL-OBJ-BALL01">')
        for i, f in enumerate(range(first, last + 1)):
            bx, by = m.ball[f]
            lab = m.y[f]
            status = 0 if lab == D else 1
            # on dead frames the provider still names a team; use the half number to vary it
            poss = 1 if lab == H else 2 if lab == A else (1 if f % 2 else 2)
            lines.append(f'<Frame N="{n0 + i}" T="2022-10-15T11:0{half}:00.000+00:00" '
                         f'X="{bx:.2f}" Y="{-by:.2f}" Z="0.10" S="1.00" M="1" '
                         f'BallPossession="{poss}" BallStatus="{status}"/>')
        lines.append("</FrameSet>")
    lines += ["</Positions>", "</PutDataRequest>"]

    left = {h: (DFL_HOME if m.attack_dir[h] == 1 else DFL_AWAY) for h in m.attack_dir}
    right = {h: (DFL_AWAY if left[h] == DFL_HOME else DFL_HOME) for h in left}
    events = ['<?xml version="1.0" encoding="UTF-8"?>', "<PutDataRequest>"]
    for half in sorted(left):
        events.append(f'<Event EventId="{half}" EventTime="t"><KickOff TeamLeft="{left[half]}" '
                      f'TeamRight="{right[half]}" GameSection="{DFL_SECTIONS[half]}"/></Event>')
    events.append("</PutDataRequest>")

    paths = {"meta": folder / "meta.xml", "tracking": folder / "tracking.xml",
             "event": folder / "events.xml"}
    paths["meta"].write_text(meta, encoding="utf-8")
    paths["tracking"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    paths["event"].write_text("\n".join(events) + "\n", encoding="utf-8")
    return paths


# -- EPTS --------------------------------------------------------------------------------------

def _epts_player_xml(s, team_id):
    return (f'<Player id="P{s}" teamId="{team_id}"><Name>Player {s}</Name>'
            f'<ShirtNumber>{s + 1}</ShirtNumber><ProviderPlayerParameters>'
            f'<ProviderParameter><Name>position_index</Name><Value>1</Value></ProviderParameter>'
            f'</ProviderPlayerParameters></Player>')


def _epts_spec(start, end, order):
    groups = "".join(
        f'<SplitRegister separator=","><PlayerChannelRef playerChannelId="player{s}_x"/>'
        f'<PlayerChannelRef playerChannelId="player{s}_y"/></SplitRegister>' for s in order)
    return (f'<DataFormatSpecification endFrame="{end}" separator=":" startFrame="{start}">'
            f'<StringRegister name="frameCount"/><SplitRegister separator=";">{groups}'
            f'</SplitRegister><SplitRegister separator=","><BallChannelRef channelId="x"/>'
            f'<BallChannelRef channelId="y"/></SplitRegister></DataFormatSpecification>')


def write_epts(m, folder, pitch=(105, 68)):
    """EPTS metadata + raw TXT for the halves of `m`, frames numbered from 1, halves adjacent.

    Two DataFormatSpecifications with different slot orders, to exercise layout switching.
    Ball is written as NaN on dead frames, like Metrica's open sample.
    Returns (paths, rows): rows[i] is the row of `m` written as source frame i + 1.
    """
    folder.mkdir(parents=True, exist_ok=True)
    rows = np.concatenate([np.arange(a, b + 1) for _, a, b in m.half_blocks])
    n_src = len(rows)
    n1 = m.half_blocks[0][2] - m.half_blocks[0][1] + 1
    home_dir = "left_to_right" if m.attack_dir[1] == 1 else "right_to_left"
    away_dir = "right_to_left" if home_dir == "left_to_right" else "left_to_right"
    S = m.n_slots
    players = "".join(_epts_player_xml(s, "TA" if m.team[s] == 0 else "TB") for s in range(S))
    channels = "".join(f'<PlayerChannel channelId="x" id="player{s}_x" playerId="P{s}"/>'
                       f'<PlayerChannel channelId="y" id="player{s}_y" playerId="P{s}"/>'
                       for s in range(S))
    order_a = list(range(S))
    order_b = list(range(S))[::-1]
    split = n1 // 2
    specs = _epts_spec(1, split, order_a) + _epts_spec(split + 1, n_src, order_b)
    meta = f"""<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig>
      <TrackingType>Optical</TrackingType>
      <ProviderName>Metrica Sports</ProviderName>
      <FrameRate>{m.fps}</FrameRate>
      <ProviderGlobalParameters>
        <ProviderParameter><Name>first_half_start</Name><Value>1</Value></ProviderParameter>
        <ProviderParameter><Name>first_half_end</Name><Value>{n1}</Value></ProviderParameter>
        <ProviderParameter><Name>second_half_start</Name><Value>{n1 + 1}</Value></ProviderParameter>
        <ProviderParameter><Name>second_half_end</Name><Value>{n_src}</Value></ProviderParameter>
      </ProviderGlobalParameters>
    </GlobalConfig>
    <Sessions><Session id="s1"><SessionType>Match</SessionType>
      <MatchParameters><Score idLocalTeam="TA" idVisitingTeam="TB">
        <LocalTeamScore>0</LocalTeamScore><VisitingTeamScore>0</VisitingTeamScore>
      </Score></MatchParameters>
      <FieldSize><Width>{pitch[0]}</Width><Height>{pitch[1]}</Height></FieldSize>
    </Session></Sessions>
    <Teams>
      <Team id="TA"><Name>Team A</Name><ProviderTeamsParameters><ProviderParameter>
        <Name>attack_direction_first_half</Name><Value>{home_dir}</Value>
      </ProviderParameter></ProviderTeamsParameters></Team>
      <Team id="TB"><Name>Team B</Name><ProviderTeamsParameters><ProviderParameter>
        <Name>attack_direction_first_half</Name><Value>{away_dir}</Value>
      </ProviderParameter></ProviderTeamsParameters></Team>
    </Teams>
    <Players>{players}</Players>
    <Devices><Device id="d1"><Name>Device 1</Name><Sensors><Sensor id="position">
      <Name>Position</Name><Channels>
        <Channel id="x"><Name>position_x</Name><Unit>normalized</Unit></Channel>
        <Channel id="y"><Name>position_y</Name><Unit>normalized</Unit></Channel>
      </Channels></Sensor></Sensors></Device></Devices>
    <PlayerChannels>{channels}</PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>{specs}</DataFormatSpecifications>
</main>
"""
    L, W = pitch

    def fmt(v):
        return "NaN" if not np.isfinite(v) else f"{v:.5f}"

    out = []
    for i, r in enumerate(rows):
        frame = i + 1
        order = order_a if frame <= split else order_b
        xy01 = np.stack([m.P[r, :, 0] / L + 0.5, m.P[r, :, 1] / W + 0.5], axis=-1)
        groups = ";".join(f"{fmt(xy01[s, 0])},{fmt(xy01[s, 1])}" for s in order)
        if m.y[r] == D:
            ball = "NaN,NaN"
        else:
            ball = f"{fmt(m.ball[r, 0] / L + 0.5)},{fmt(m.ball[r, 1] / W + 0.5)}"
        out.append(f"{frame}:{groups}:{ball}")
    paths = {"meta": folder / "metadata.xml", "raw": folder / "tracking.txt"}
    paths["meta"].write_text(meta, encoding="utf-8")
    paths["raw"].write_text("\n".join(out) + "\n", encoding="ascii")
    return paths, rows
