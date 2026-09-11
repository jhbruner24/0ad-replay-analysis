"""Parse commands.txt to produce per-window action counts by player.

commands.txt is a per-turn event stream at whatever turn length the game used
(typically 200-500ms), which is far finer than metadata.json's 30-second stat
sequences. This module extracts action-rate features so the model can update
between 30-second stat snapshots without inventing information.

The file's shape (see 0ad-replay-coach/README.md for the reasoning):

    start <json attribs>
    turn <n> <turnLength_ms>
    cmd <player_id> <json>       # zero or more per turn
    cmd <player_id> <json>
    end
    hash <hex>                   # optional
    turn <n+1> ...

Player IDs in `cmd` lines are 1-indexed and align with metadata.json's
`playerStates[1:]` (playerStates[0] is Gaia).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass


# The command types worth exposing as features. Chosen because they encode
# decisions - what the player is doing - rather than being noise or system.
# Ranked roughly by frequency in real games; anything not listed is folded
# into "_other" so total-action rate stays informative.
TRACKED_TYPES = (
    "walk",           # unit movement, dominant volume, tracks APM baseline
    "gather",         # economy tempo
    "attack",         # engagement rate
    "train",          # production tempo
    "research",       # tech investment
    "construct",      # building placement
    "construct-wall",
    "garrison",
    "repair",
    "set-rallypoint",
    "unload-template",
    "unload-all",
    "returnresource",
    "barter",
    "heal",
    "stance",
    "alert-raise",
    "delete-entities",
    "call-to-arms",
    "resign",         # rare but decisive - game-ending signal
)


@dataclass
class Event:
    t: float          # seconds since game start
    player_id: int    # 1-indexed
    type: str


def parse(commands_path: str, max_events: int | None = None) -> list[Event]:
    """Parse one commands.txt into a flat list of events.

    Never raises on malformed lines; skips them. Turn lengths accumulate from
    the file itself, so if the game changed speed mid-run we still get the
    right times.
    """
    events: list[Event] = []
    t_ms = 0.0
    try:
        with open(commands_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line or line[0] == "s":              # `start` line
                    continue
                if line[0] == "t":                          # turn <n> <length>
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            t_ms += float(parts[2])
                        except ValueError:
                            pass
                elif line[0] == "c":                        # cmd <pid> <json>
                    parts = line.split(None, 2)
                    if len(parts) < 3:
                        continue
                    try:
                        pid = int(parts[1])
                        cmd = json.loads(parts[2])
                    except (ValueError, json.JSONDecodeError):
                        continue
                    ctype = cmd.get("type") if isinstance(cmd, dict) else None
                    if not ctype:
                        continue
                    events.append(Event(t=t_ms / 1000.0, player_id=pid, type=ctype))
                    if max_events and len(events) >= max_events:
                        break
    except OSError:
        pass
    return events


def counts_in_window(events, player_id: int, t_start: float, t_end: float) -> Counter:
    """Actions by type for one player, inside [t_start, t_end).

    Buckets anything outside TRACKED_TYPES as `_other` so total-action rate is
    still meaningful even when the game does something we didn't anticipate.
    """
    tracked = set(TRACKED_TYPES)
    out: Counter = Counter()
    for e in events:
        if e.player_id != player_id:
            continue
        if e.t < t_start or e.t >= t_end:
            continue
        out[e.type if e.type in tracked else "_other"] += 1
    return out


# Higher-level categories. A raw count of "walk" commands is dominated by
# routine unit movement; grouping tells the model whether the player is in
# economy, army, or defensive mode.
CATEGORIES = {
    "economy":     ("gather", "returnresource", "barter", "construct",
                    "repair", "set-rallypoint"),
    "military":    ("attack", "walk", "garrison", "unload-template",
                    "unload-all", "stance", "call-to-arms", "heal"),
    "production":  ("train", "research", "stop-production"),
    "defensive":   ("construct-wall", "alert-raise", "alert-end", "garrison"),
    "surrender":   ("resign",),
}


def category_counts(counts: Counter) -> dict:
    """Sum tracked-type counts into the higher-level categories."""
    out = {}
    for cat, types in CATEGORIES.items():
        out[cat] = sum(counts.get(t, 0) for t in types)
    return out


def shannon_entropy(counts: Counter) -> float:
    """Diversity of action mix, natural log. 0 = only one type; high = varied.

    Two players issuing the same number of total commands can have very
    different distributions: one spamming `walk`, one juggling gather, train,
    and attack. That distinction is invisible to per-type counts alone.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    from math import log
    return -sum((c / total) * log(c / total) for c in counts.values() if c > 0)


def rate_acceleration(events, player_id: int, t: float,
                      short: float = 10, long: float = 30) -> float:
    """Actions/sec in the last `short` seconds minus in the preceding `long`.

    A positive value flags a burst - e.g. an incoming attack triggers many
    fresh commands in a small window. Negative values flag lulls.
    """
    if t < long + short:
        return 0.0
    n_short = sum(1 for e in events if e.player_id == player_id
                  and t - short <= e.t < t)
    n_long = sum(1 for e in events if e.player_id == player_id
                 and t - short - long <= e.t < t - short)
    return n_short / short - n_long / long


def parse_for_game(game_directory: str) -> list[Event]:
    """Convenience for pairing with a `schema.Game`."""
    return parse(os.path.join(game_directory, "commands.txt"))
