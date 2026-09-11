"""Generate a synthetic replay collection for development and testing.

Writes real directory structures with real-shaped metadata.json, so the rest of
the package can be built and tested before meeting an actual collection.

Deliberately messy by default, because real collections are: some games have no
metadata.json (crashes), some are unresolved (abandoned), engine versions are
mixed, game lengths vary, and the odd game is a team game rather than a 1v1.

It also plants a *known* signal - the winner gathers food faster in a specific
mid-game window - so analysis code can be checked against a ground truth it is
supposed to recover.

    python3 -m oadrep.synth /tmp/fake-replays --games 200 --drift 0.3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

SIGNAL_WINDOW = (300, 600)          # seconds; where the planted advantage lives
SIGNAL_STRENGTH = 0.45              # relative food-rate edge for the winner


def _series(n, rate, noise, rng, curve=1.0):
    """A cumulative, monotonically rising counter - what the game actually logs."""
    out, total = [], 0.0
    for i in range(n):
        step = rate * (1 + curve * i / max(n, 1)) * rng.uniform(1 - noise, 1 + noise)
        total += max(step, 0.0)
        out.append(round(total))
    return out


def _instant(n, base, spread, rng):
    """A level, not a counter - population, stockpiles, percentages."""
    return [max(0, round(base + spread * math.sin(i / 6) + rng.uniform(-spread, spread)))
            for i in range(n)]


UNIT_CLASSES = ("total", "Infantry", "Cavalry", "Champion", "Siege", "Ship",
                "Hero", "Worker", "Trader", "Unit")
BUILDING_CLASSES = ("total", "Economic", "CivCentre", "Fortress", "House",
                    "Military", "Outpost", "Structure", "Wonder")


def _by_class(n, classes, rate, rng, total=True):
    out = {}
    for c in classes:
        if c == "total" and not total:
            out[c] = [0] * n
        else:
            out[c] = _series(n, rate * (1.0 if c in ("total", "Unit", "Structure", "Worker") else 0.3), 0.4, rng)
    return out


def _player(name, civ, won, snapshots, rng):
    times = [30.0 * (k + 1) for k in range(snapshots)]

    # The planted signal: the winner gathers food faster, but only mid-game.
    food = []
    total = 0.0
    for t in times:
        rate = 95 * rng.uniform(0.85, 1.15)
        if won and SIGNAL_WINDOW[0] <= t <= SIGNAL_WINDOW[1]:
            rate *= 1 + SIGNAL_STRENGTH
        total += rate
        food.append(round(total))

    return {
        "name": name,
        "civ": civ,
        "state": "won" if won else "defeated",
        "phase": "phase_city" if won else "phase_town",
        "popCount": rng.randint(80, 160),
        "popLimit": 200,
        "resourceGatherers": {"food": rng.randint(12, 30), "wood": rng.randint(8, 25)},
        "classCounts": {"Infantry": rng.randint(20, 60), "Cavalry": rng.randint(2, 25)},
        "typeCountsByClass": {},
        "researchedTechs": ["phase_town", "phase_city"] if won else ["phase_town"],
        "team": -1,
        "sequences": {
            "time": times,
            "populationCount": _instant(snapshots, 60 if won else 45, 12, rng),
            "percentMapControlled": _instant(snapshots, 22 if won else 16, 4, rng),
            "percentMapExplored": _instant(snapshots, 40, 10, rng),
            # Per-class counters as v0.27+ writes them. The engine never
            # increments the `total` bucket of the *Lost counters; mirror that.
            "unitsTrained": _by_class(snapshots, UNIT_CLASSES, 2.0, rng),
            "unitsLost": _by_class(snapshots, UNIT_CLASSES, 0.7 if won else 1.1, rng, total=False),
            "unitsCaptured": _by_class(snapshots, UNIT_CLASSES, 0.02, rng),
            "enemyUnitsKilled": _by_class(snapshots, UNIT_CLASSES, 1.1 if won else 0.7, rng),
            "unitsLostValue": _series(snapshots, 80 if won else 120, 0.5, rng),
            "buildingsConstructed": _by_class(snapshots, BUILDING_CLASSES, 0.4, rng),
            "buildingsLost": _by_class(snapshots, BUILDING_CLASSES, 0.05, rng, total=False),
            "buildingsCaptured": _by_class(snapshots, BUILDING_CLASSES, 0.01, rng),
            "buildingsLostValue": _series(snapshots, 10, 0.5, rng),
            "tradeIncome": _series(snapshots, 5, 0.5, rng),
            "resourcesGathered": {
                "food": food,
                "wood": _series(snapshots, 70, 0.25, rng),
                "stone": _series(snapshots, 25, 0.4, rng),
                "metal": _series(snapshots, 30, 0.4, rng),
            },
            "resourcesUsed": {
                "food": _series(snapshots, 60, 0.3, rng),
                "wood": _series(snapshots, 55, 0.3, rng),
                "stone": _series(snapshots, 20, 0.4, rng),
                "metal": _series(snapshots, 25, 0.4, rng),
            },
            "resourcesCount": {
                "food": _instant(snapshots, 400, 250, rng),
                "wood": _instant(snapshots, 350, 200, rng),
            },
        },
    }


def generate(root, games=200, me="Me", opponents=("Friend",), base_win_rate=0.35,
             drift=0.0, missing_metadata=0.04, team_games=0.02,
             unresolved=0.03, versions=("0.27.0",), rated=0.6,
             missing_version=0.25, left_running=0.005, seed=1):
    """Write `games` synthetic replays under `root`. Returns the manifest.

    Defaults mirror the messiness of a real collection: lobby ratings appended to
    nicknames, some replays with no engine_version in the start line, and the
    occasional game left running for hours.
    """
    rng = random.Random(seed)
    os.makedirs(root, exist_ok=True)
    manifest = []
    if isinstance(opponents, str):
        opponents = (opponents,)
    # A stable "true skill" per opponent, so rating carries real signal.
    skill = {name: rng.uniform(-0.25, 0.25) for name in opponents}

    for i in range(games):
        # Win rate walks from base_win_rate to base_win_rate+drift across the
        # collection, so drift-detection code has something real to find.
        progress = i / max(games - 1, 1)
        opponent = rng.choice(opponents)
        p_win = min(max(base_win_rate + drift * progress - skill[opponent], 0.02), 0.98)
        i_won = rng.random() < p_win

        # A9: lobby games carry "nick (rating)"; ratings track the skill above.
        if rng.random() < rated:
            my_name = f"{me} ({rng.randint(1150, 1450)})"
            their_name = f"{opponent} ({int(1300 + skill[opponent] * 600 + rng.randint(-40, 40))})"
        else:
            my_name, their_name = me, opponent

        version = versions[min(int(progress * len(versions)), len(versions) - 1)]
        directory = os.path.join(root, version, f"{1700000000 + i * 7200}_{i:04d}")
        os.makedirs(directory, exist_ok=True)

        snapshots = rng.randint(14, 60)          # 7 to 30 minutes at 30s steps
        if rng.random() < left_running:          # game left running, not played
            snapshots = rng.randint(400, 1800)
        players = [
            _player(my_name, "athen", i_won, snapshots, rng),
            _player(their_name, rng.choice(["spart", "cart", "ptol"]),
                    not i_won, snapshots, rng),
        ]
        if rng.random() < team_games:            # occasional non-1v1
            players.append(_player("Ally", "brit", i_won, snapshots, rng))
        if rng.random() < unresolved:            # abandoned: nobody resolved
            for p in players:
                p["state"] = "active"

        # A2: the real field is settings.mapName, and older replays can omit
        # engine_version entirely - the directory name is then the only source.
        attribs = {
            "settings": {"mapName": rng.choice(["Mainland", "Arcadia", "Corsica"])},
            "mapType": "random",
            "mods": [],
        }
        if rng.random() >= missing_version:
            attribs["engine_version"] = version
        with open(os.path.join(directory, "commands.txt"), "w", encoding="utf-8") as fh:
            fh.write("start " + json.dumps(attribs) + "\n")
            for turn in range(snapshots * 60):   # 30s of 500ms turns per snapshot
                fh.write(f"turn {turn} 500\nend\n")

        wrote_meta = rng.random() >= missing_metadata
        if wrote_meta:
            meta = {
                "timeElapsed": snapshots * 30 * 1000,
                "mapSettings": {"Name": "Mainland"},
                "playerStates": [{"name": "Gaia", "civ": "gaia"}] + players,
            }
            with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh)

        manifest.append({
            "directory": directory, "i_won": i_won, "players": len(players),
            "has_metadata": wrote_meta, "version": version, "snapshots": snapshots,
            "opponent": opponent,
        })

    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--me", default="Me")
    ap.add_argument("--opponents", default="Friend",
                    help="comma-separated opponent nicks")
    ap.add_argument("--win-rate", type=float, default=0.35, help="starting win rate")
    ap.add_argument("--drift", type=float, default=0.0,
                    help="how much the win rate moves across the collection")
    ap.add_argument("--versions", default="0.27.0", help="comma-separated")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    manifest = generate(
        args.root, games=args.games, me=args.me,
        opponents=tuple(args.opponents.split(",")),
        base_win_rate=args.win_rate, drift=args.drift,
        versions=tuple(args.versions.split(",")), seed=args.seed,
    )
    wins = sum(m["i_won"] for m in manifest)
    print(f"Wrote {len(manifest)} replays to {args.root}")
    print(f"  {args.me} won {wins} ({wins/len(manifest):.1%})")
    print(f"  opponents: {len(set(m['opponent'] for m in manifest))}")
    print(f"  missing metadata: {sum(not m['has_metadata'] for m in manifest)}")
    print(f"  non-1v1:          {sum(m['players'] != 2 for m in manifest)}")
    print(f"\nPlanted signal: winners gather food ~{SIGNAL_STRENGTH:.0%} faster "
          f"between {SIGNAL_WINDOW[0]}s and {SIGNAL_WINDOW[1]}s.")


if __name__ == "__main__":
    main()
