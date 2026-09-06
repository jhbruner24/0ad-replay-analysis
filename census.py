#!/usr/bin/env python3
"""Inventory the local 0 A.D. replay collection.

Answers, before any modelling starts: how many games are actually usable, are
they all the same matchup, do they span engine versions, and has the head-to-head
win rate drifted over time?

Reads only metadata.json and the `start` line of commands.txt. Does not run the
engine. Read-only; never writes into the replay directory.

Usage:
    python3 census.py                      # summary
    python3 census.py --me "YourName"      # adds win/loss and drift analysis
    python3 census.py --csv out.csv        # also flatten sequences to long-format CSV

If --me is omitted the script prints the player names it saw so you can pick one.

NOTE: written against the a27-era engine source and not yet tested against a real
replay collection. Field names may differ on v28; --debug dumps the raw keys.
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict

DEFAULT_ROOTS = [
    "~/.local/share/0ad/replays",                                  # Linux
    "~/Library/Application Support/0ad/replays",                   # macOS
    "~/AppData/Roaming/0ad/replays",                               # Windows
]


def find_root(explicit):
    if explicit:
        p = os.path.expanduser(explicit)
        return p if os.path.isdir(p) else None
    for cand in DEFAULT_ROOTS:
        p = os.path.expanduser(cand)
        if os.path.isdir(p):
            return p
    return None


def load_json(path):
    """Tolerate truncated or malformed files - a crashed game can leave either."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_start_attribs(commands_path):
    """The first line is `start <json>`; it carries engine version and mods."""
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


def count_turns(commands_path):
    """Cheap line scan; turn count stands in for game length in sim terms."""
    turns = 0
    try:
        with open(commands_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("turn "):
                    turns += 1
    except OSError:
        return None
    return turns


def sort_key(directory):
    """Replay dirs are named <unixtime>_<n>; fall back to mtime."""
    m = re.match(r"(\d+)", os.path.basename(directory))
    if m:
        return int(m.group(1))
    try:
        return int(os.path.getmtime(directory))
    except OSError:
        return 0


def collect(root):
    games = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "commands.txt" not in filenames:
            continue
        dirnames[:] = []  # a replay dir has no replay children
        meta = load_json(os.path.join(dirpath, "metadata.json")) if "metadata.json" in filenames else None
        attribs = read_start_attribs(os.path.join(dirpath, "commands.txt"))
        games.append({
            "dir": dirpath,
            "order": sort_key(dirpath),
            "meta": meta,
            "attribs": attribs,
        })
    games.sort(key=lambda g: g["order"])
    return games


def players_of(meta):
    """playerStates[0] is Gaia; real players start at index 1."""
    if not isinstance(meta, dict):
        return []
    states = meta.get("playerStates")
    if not isinstance(states, list):
        return []
    return [p for p in states[1:] if isinstance(p, dict)]


def histogram(title, counter, limit=12):
    print(f"\n{title}")
    if not counter:
        print("  (none)")
        return
    width = max(len(str(k)) for k, _ in counter.most_common(limit))
    for key, n in counter.most_common(limit):
        print(f"  {str(key):<{width}}  {n}")
    if len(counter) > limit:
        print(f"  ... and {len(counter) - limit} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", help="replay directory (auto-detected if omitted)")
    ap.add_argument("--me", help="your in-game player name, for win/loss and drift")
    ap.add_argument("--csv", help="write long-format sequence data here")
    ap.add_argument("--blocks", type=int, default=5, help="chronological blocks for drift (default 5)")
    ap.add_argument("--debug", action="store_true", help="dump raw keys of the first game")
    args = ap.parse_args()

    root = find_root(args.root)
    if not root:
        sys.exit("No replay directory found. Pass --root /path/to/replays")
    print(f"Replay root: {root}")

    games = collect(root)
    if not games:
        sys.exit("No replays found (looked for directories containing commands.txt).")

    with_meta = [g for g in games if g["meta"]]
    print(f"\nReplay directories:        {len(games)}")
    print(f"  with usable metadata.json: {len(with_meta)}")
    print(f"  without (unfinished?):     {len(games) - len(with_meta)}")

    if args.debug and with_meta:
        print("\n--- debug: first game ---")
        print("metadata keys:", sorted(with_meta[0]["meta"].keys()))
        ps = players_of(with_meta[0]["meta"])
        if ps:
            print("player keys:", sorted(ps[0].keys()))
            seq = ps[0].get("sequences")
            if isinstance(seq, dict):
                print("sequence keys:", sorted(seq.keys()))
                t = seq.get("time")
                print("sequence length:", len(t) if isinstance(t, list) else "n/a")

    versions, sizes, maps, matchups, names = Counter(), Counter(), Counter(), Counter(), Counter()
    durations, seq_lens, one_v_one = [], [], []

    for g in with_meta:
        ps = players_of(g["meta"])
        sizes[len(ps)] += 1
        for p in ps:
            if p.get("name"):
                names[p["name"]] += 1

        if g["attribs"]:
            versions[g["attribs"].get("engine_version", "unknown")] += 1
            settings = g["attribs"].get("settings") or {}
            maps[settings.get("Name") or g["attribs"].get("mapType", "unknown")] += 1

        elapsed = g["meta"].get("timeElapsed")
        if isinstance(elapsed, (int, float)):
            durations.append(elapsed / 60000.0)  # ms -> minutes

        if len(ps) == 2:
            one_v_one.append(g)
            matchups[" vs ".join(sorted(str(p.get("civ", "?")) for p in ps))] += 1
            for p in ps:
                seq = p.get("sequences")
                if isinstance(seq, dict) and isinstance(seq.get("time"), list):
                    seq_lens.append(len(seq["time"]))

    print(f"\n1v1 games (2 players):     {len(one_v_one)}")
    histogram("Players per game", sizes)
    histogram("Engine versions", versions)
    histogram("Player names", names)
    histogram("Civ matchups", matchups)
    histogram("Maps", maps)

    if durations:
        durations.sort()
        print(f"\nGame length (minutes), n={len(durations)}")
        print(f"  median {statistics.median(durations):.1f}   "
              f"p10 {durations[len(durations)//10]:.1f}   "
              f"p90 {durations[-max(1, len(durations)//10)]:.1f}")
    if seq_lens:
        print(f"\nSequence snapshots per player: median {statistics.median(seq_lens):.0f}, "
              f"min {min(seq_lens)}, max {max(seq_lens)}")
        print("  (one snapshot per 30s of game time -> this is your time resolution)")
    else:
        print("\nWARNING: no `sequences` found in any player state.")
        print("  Time-series analysis is not possible from metadata.json alone.")

    if not args.me:
        print("\nRe-run with --me \"<name>\" (from the Player names list) for win/loss and drift.")
    else:
        mine = []
        for g in one_v_one:
            ps = players_of(g["meta"])
            me = next((p for p in ps if p.get("name") == args.me), None)
            them = next((p for p in ps if p.get("name") != args.me), None)
            if me and them:
                mine.append((g, me, them))

        print(f"\n=== Head-to-head for {args.me}: {len(mine)} games ===")
        if not mine:
            print("  No games matched that name.")
        else:
            opponents = Counter(t.get("name", "?") for _, _, t in mine)
            histogram("Opponents", opponents)

            outcomes = Counter(str(me.get("state", "unknown")) for _, me, _ in mine)
            histogram("Your outcomes", outcomes)

            decided = [(g, me, t) for g, me, t in mine
                       if str(me.get("state")) in ("won", "defeated")]
            if decided:
                wins = sum(1 for _, me, _ in decided if str(me.get("state")) == "won")
                rate = wins / len(decided)
                print(f"\nOverall win rate: {wins}/{len(decided)} = {rate:.1%}")
                print(f"  Baseline to beat: {max(rate, 1 - rate):.1%} "
                      "(always predicting the more frequent winner)")

                n = args.blocks
                size = max(1, len(decided) // n)
                print(f"\nDrift check - win rate in {n} chronological blocks:")
                for i in range(n):
                    chunk = decided[i * size:(i + 1) * size] if i < n - 1 else decided[i * size:]
                    if not chunk:
                        continue
                    w = sum(1 for _, me, _ in chunk if str(me.get("state")) == "won")
                    bar = "#" * round(20 * w / len(chunk))
                    print(f"  block {i+1}: {w:>3}/{len(chunk):<3} {w/len(chunk):>6.1%} |{bar}")
                print("\n  Flat -> matchup is stable, pool all games.")
                print("  Trending -> weight recent games; split temporally, never randomly.")

    if args.csv:
        rows = 0
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["game", "order", "player", "civ", "state", "t_seconds", "stat", "value"])
            for g in one_v_one:
                gid = os.path.basename(g["dir"])
                for p in players_of(g["meta"]):
                    seq = p.get("sequences")
                    if not isinstance(seq, dict) or not isinstance(seq.get("time"), list):
                        continue
                    times = seq["time"]
                    for stat, series in seq.items():
                        if stat == "time":
                            continue
                        # Resource stats nest one level: {food: [...], wood: [...]}
                        cols = (series if isinstance(series, dict)
                                else {"": series} if isinstance(series, list) else {})
                        for sub, vals in cols.items():
                            if not isinstance(vals, list):
                                continue
                            label = f"{stat}.{sub}" if sub else stat
                            for t, v in zip(times, vals):
                                if isinstance(v, (int, float)):
                                    w.writerow([gid, g["order"], p.get("name", "?"),
                                                p.get("civ", "?"), p.get("state", "?"),
                                                t, label, v])
                                    rows += 1
        print(f"\nWrote {rows} rows to {args.csv}")


if __name__ == "__main__":
    main()
