"""The single place that knows the shape of 0 A.D. replay files.

Every assumption about metadata.json and commands.txt lives here. Nothing else in
this package reads those files directly, so if a release renames a field, this is
the only module that changes.

Assumptions, all read from engine source (2024 GitHub mirror, roughly a27) and
flagged individually below so they can be checked one at a time against a real
collection:

  A1  A replay is a directory containing commands.txt, usually beside metadata.json.
  A2  commands.txt line 1 is `start <json>`, carrying engine_version and settings.
  A3  metadata.json has keys timeElapsed (ms), playerStates, mapSettings.
  A4  playerStates[0] is Gaia; real players start at index 1.
  A5  Each player has name, civ, state ("won"/"defeated"/"active").
  A6  player["sequences"] is column-oriented: {"time": [...], "<stat>": [...]},
      with resource-typed stats nested one level: {"food": [...], "wood": [...]}.
  A7  sequences["time"] is in SECONDS; timeElapsed is in MILLISECONDS.
  A8  Replay directories are named "<unixtime>_<n>", giving chronological order.

Run `python3 -m oadrep.schema <replay-dir>` to check these against a real game.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

WON, DEFEATED = "won", "defeated"


@dataclass
class Player:
    name: str
    civ: str
    state: str
    sequences: dict          # flattened: {"resourcesGathered.food": [...], ...}
    times: list              # seconds, parallel to every series in `sequences`
    snapshot: dict           # end-of-game only: phase, researchedTechs, ...

    @property
    def won(self) -> bool | None:
        if self.state == WON:
            return True
        if self.state == DEFEATED:
            return False
        return None          # unresolved: game abandoned or still "active"


@dataclass
class Game:
    directory: str
    order: int               # sortable; chronological
    engine_version: str
    map_name: str
    duration_minutes: float | None
    players: list[Player] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def is_1v1(self) -> bool:
        return len(self.players) == 2

    @property
    def decided(self) -> bool:
        return self.is_1v1 and sum(p.won is True for p in self.players) == 1

    def perspective(self, name: str) -> tuple[Player, Player] | None:
        """Return (me, opponent) for a 1v1, or None if `name` isn't in it."""
        if not self.is_1v1:
            return None
        me = next((p for p in self.players if p.name == name), None)
        them = next((p for p in self.players if p is not me), None)
        return (me, them) if me and them else None


# A6: flatten one level of nesting so every stat is a plain named series.
def flatten_sequences(raw) -> dict[str, list]:
    out = {}
    if not isinstance(raw, dict):
        return out
    for stat, series in raw.items():
        if stat == "time":
            continue
        if isinstance(series, list):
            out[stat] = series
        elif isinstance(series, dict):
            for sub, vals in series.items():
                if isinstance(vals, list):
                    out[f"{stat}.{sub}"] = vals
    return out


SNAPSHOT_KEYS = (
    "phase", "popCount", "popLimit", "resourceGatherers",
    "classCounts", "typeCountsByClass", "researchedTechs", "team",
)


def _load_json(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _start_attribs(commands_path):
    """A2. Only the first line is read; commands.txt can be enormous."""
    try:
        with open(commands_path, encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
    except OSError:
        return None
    if not first.startswith("start "):
        return None
    try:
        return json.loads(first[len("start "):])
    except ValueError:
        return None


def _order_of(directory) -> int:
    m = re.match(r"(\d+)", os.path.basename(directory))       # A8
    if m:
        return int(m.group(1))
    try:
        return int(os.path.getmtime(directory))
    except OSError:
        return 0


def load_game(directory: str) -> Game:
    """Parse one replay directory. Never raises; unreadable parts become problems."""
    problems = []
    meta = _load_json(os.path.join(directory, "metadata.json"))
    attribs = _start_attribs(os.path.join(directory, "commands.txt"))

    if meta is None:
        problems.append("no usable metadata.json (unfinished or crashed game)")
    if attribs is None:
        problems.append("no usable start line in commands.txt")

    engine = (attribs or {}).get("engine_version", "unknown")
    settings = (attribs or {}).get("settings") or {}
    map_name = settings.get("Name") or (meta or {}).get("mapSettings", {}).get("Name") or "unknown"

    duration = None
    elapsed = (meta or {}).get("timeElapsed")
    if isinstance(elapsed, (int, float)):
        duration = elapsed / 60000.0                          # A7

    players = []
    states = (meta or {}).get("playerStates")
    if isinstance(states, list):
        for entry in states[1:]:                              # A4
            if not isinstance(entry, dict):
                continue
            seqs = entry.get("sequences")
            times = seqs.get("time") if isinstance(seqs, dict) else None
            if not isinstance(times, list):
                times = []
            players.append(Player(
                name=str(entry.get("name", "?")),
                civ=str(entry.get("civ", "?")),
                state=str(entry.get("state", "unknown")),     # A5
                sequences=flatten_sequences(seqs),
                times=times,
                snapshot={k: entry[k] for k in SNAPSHOT_KEYS if k in entry},
            ))
    elif meta is not None:
        problems.append("metadata.json has no playerStates list")

    if players and not any(p.times for p in players):
        problems.append("no `sequences` present: no time-series data available")

    return Game(
        directory=directory, order=_order_of(directory), engine_version=engine,
        map_name=map_name, duration_minutes=duration, players=players,
        problems=problems,
    )


def find_root(explicit=None):
    candidates = [explicit] if explicit else [
        "~/.local/share/0ad/replays",                    # Linux
        "~/Library/Application Support/0ad/replays",     # macOS
        "~/AppData/Roaming/0ad/replays",                 # Windows
    ]
    for cand in candidates:
        if cand:
            path = os.path.expanduser(cand)
            if os.path.isdir(path):
                return path
    return None


def iter_games(root: str):
    """Yield every replay directory under `root`, in chronological order. (A1)"""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "commands.txt" in filenames:
            dirnames[:] = []          # replay dirs do not nest
            found.append(dirpath)
    for directory in sorted(found, key=_order_of):
        yield load_game(directory)


def check(directory: str) -> int:
    """Verify the documented assumptions against one real replay directory."""
    game = load_game(directory)
    print(f"Directory : {game.directory}")
    print(f"A8 order  : {game.order}")
    print(f"A2 engine : {game.engine_version}")
    print(f"A3 map    : {game.map_name}")
    print(f"A7 length : {game.duration_minutes:.1f} min"
          if game.duration_minutes is not None else "A7 length : MISSING")
    print(f"A4 players: {len(game.players)}")
    for p in game.players:
        print(f"   - {p.name:<16} civ={p.civ:<12} state={p.state:<10} "
              f"snapshots={len(p.times):<4} stats={len(p.sequences)}")
        if p.times:
            step = p.times[1] - p.times[0] if len(p.times) > 1 else float("nan")
            print(f"     A6/A7 first times={p.times[:3]} step={step}")
        missing = [k for k in SNAPSHOT_KEYS if k not in p.snapshot]
        if missing:
            print(f"     A5 snapshot keys absent: {', '.join(missing)}")
    if game.problems:
        print("\nPROBLEMS:")
        for prob in game.problems:
            print(f"  - {prob}")
        return 1
    print("\nAll assumptions hold for this game.")
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        sys.exit("usage: python3 -m oadrep.schema <replay-directory>")
    sys.exit(check(sys.argv[1]))
