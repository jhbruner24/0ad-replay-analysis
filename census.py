#!/usr/bin/env python3
"""Inventory a 0 A.D. replay collection.

Answers, before any modelling starts: how many games are actually usable, are
they all the same matchup, do they span engine versions, and has the head-to-head
win rate drifted over time?

Reads only metadata.json and the `start` line of commands.txt. Does not run the
engine. Strictly read-only; never writes into the replay directory.

    python3 census.py                      # summary
    python3 census.py --me "YourName"      # adds win/loss and drift analysis
    python3 census.py --csv out.csv        # also flatten sequences to long CSV

If --me is omitted the script prints the player names it saw so you can pick one.
All knowledge of file formats lives in oadrep/schema.py; --debug there checks the
documented assumptions against a single replay.
"""

import argparse
import csv
import os
import statistics
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oadrep import schema


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
    ap.add_argument("--blocks", type=int, default=5, help="chronological blocks for drift")
    ap.add_argument("--debug", action="store_true", help="dump schema details of the first game")
    args = ap.parse_args()

    root = schema.find_root(args.root)
    if not root:
        sys.exit("No replay directory found. Pass --root /path/to/replays")
    print(f"Replay root: {root}")

    games = list(schema.iter_games(root))
    if not games:
        sys.exit("No replays found (looked for directories containing commands.txt).")

    usable = [g for g in games if g.players]
    print(f"\nReplay directories:          {len(games)}")
    print(f"  with usable player data:   {len(usable)}")
    print(f"  without (unfinished?):     {len(games) - len(usable)}")

    problems = Counter(p for g in games for p in g.problems)
    if problems:
        histogram("Problems encountered", problems)

    if args.debug and usable:
        print("\n--- schema check on first usable game ---")
        schema.check(usable[0].directory)

    versions, sizes, maps, matchups, names = Counter(), Counter(), Counter(), Counter(), Counter()
    durations, seq_lens, one_v_one = [], [], []

    for g in usable:
        sizes[len(g.players)] += 1
        versions[g.engine_version] += 1
        maps[g.map_name] += 1
        for p in g.players:
            names[p.name] += 1
        if g.duration_minutes is not None:
            durations.append(g.duration_minutes)
        if g.is_1v1:
            one_v_one.append(g)
            matchups[" vs ".join(sorted(p.civ for p in g.players))] += 1
            seq_lens.extend(len(p.times) for p in g.players if p.times)

    print(f"\n1v1 games (2 players):       {len(one_v_one)}")
    print(f"  of which decided:          {sum(g.decided for g in one_v_one)}")
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
        print("  Run `python3 -m oadrep.schema <a replay dir>` to see what is there.")

    if not args.me:
        print("\nRe-run with --me \"<name>\" (from the Player names list) "
              "for win/loss and drift.")
    else:
        mine = [(g, *g.perspective(args.me)) for g in one_v_one
                if g.perspective(args.me) and g.perspective(args.me)[0]]
        print(f"\n=== Head-to-head for {args.me}: {len(mine)} games ===")
        if not mine:
            print("  No games matched that name.")
        else:
            histogram("Opponents", Counter(t.name for _, _, t in mine))
            histogram("Your outcomes", Counter(me.state for _, me, _ in mine))

            decided = [(g, me, t) for g, me, t in mine if me.won is not None]
            if decided:
                wins = sum(1 for _, me, _ in decided if me.won)
                rate = wins / len(decided)
                print(f"\nOverall win rate: {wins}/{len(decided)} = {rate:.1%}")
                print(f"  Baseline to beat: {max(rate, 1 - rate):.1%} "
                      "(always predicting the more frequent winner)")

                n = args.blocks
                size = max(1, len(decided) // n)
                print(f"\nDrift check - win rate in {n} chronological blocks:")
                for i in range(n):
                    chunk = decided[i*size:(i+1)*size] if i < n-1 else decided[i*size:]
                    if not chunk:
                        continue
                    w = sum(1 for _, me, _ in chunk if me.won)
                    bar = "#" * round(20 * w / len(chunk))
                    print(f"  block {i+1}: {w:>3}/{len(chunk):<3} {w/len(chunk):>6.1%} |{bar}")
                print("\n  Flat -> matchup is stable, pool all games.")
                print("  Trending -> weight recent games; split temporally, never randomly.")

    if args.csv:
        rows = 0
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["game", "order", "player", "civ", "state",
                        "t_seconds", "stat", "value"])
            for g in one_v_one:
                gid = os.path.basename(g.directory)
                for p in g.players:
                    for stat, series in p.sequences.items():
                        for t, v in zip(p.times, series):
                            if isinstance(v, (int, float)):
                                w.writerow([gid, g.order, p.name, p.civ, p.state,
                                            t, stat, v])
                                rows += 1
        print(f"\nWrote {rows} rows to {args.csv}")


if __name__ == "__main__":
    main()
