#!/usr/bin/env python3
"""Fit the live model on the whole corpus and write it into the coach-overlay mod.

The mod scores the game itself (no Python process during play); it just needs
the fitted numbers. Re-run this whenever the replay collection has grown.

  python3 export_model.py --me wace8000 --opp wolfmapking,noob5layer --install

Defaults match live_coach.py: 15-second training grid, 200-game recency
halflife, position-only (mirrored, no rating term). --with-skill keeps the
matchup prior instead.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oadrep import model, schema

HERE = os.path.dirname(os.path.abspath(__file__))
MOD_SRC = os.path.join(HERE, "mod", "coach-overlay")
MODEL_REL = os.path.join("gui", "coach-overlay", "model.json")


def mods_dir() -> str:
    home = os.path.expanduser("~")
    if platform.system() == "Darwin":
        return f"{home}/Library/Application Support/0ad/mods"
    if platform.system() == "Windows":
        return f"{home}/AppData/Roaming/0ad/mods"
    return f"{home}/.local/share/0ad/mods"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", required=True, help="comma-separated aliases")
    ap.add_argument("--opp", default="", help="comma-separated opponent aliases")
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--halflife", type=int, default=200)
    ap.add_argument("--with-skill", action="store_true")
    ap.add_argument("--root")
    ap.add_argument("--out", default=os.path.join(MOD_SRC, MODEL_REL))
    ap.add_argument("--install", action="store_true",
                    help="also copy the whole mod into the 0 A.D. mods folder")
    args = ap.parse_args()

    root = schema.find_root(args.root)
    if not root:
        sys.exit("No replay directory found. Pass --root explicitly.")
    me = [n.strip() for n in args.me.split(",") if n.strip()]
    opp = {n.strip() for n in args.opp.split(",") if n.strip()} or None

    print("Loading historical replays...", flush=True)
    games = list(schema.iter_games(root))
    samples = model.build_samples(games, me, opp_aliases=opp, step=args.step,
                                  load_commands=True)
    if not samples:
        sys.exit("No training samples produced.")
    n_games = len({s.game_dir for s in samples})
    if not args.with_skill:
        samples = model.position_only(samples)
    print(f"  fitting on {n_games} games ({len(samples)} samples"
          f"{'' if args.with_skill else ', position-only'})...", flush=True)
    fit, names = model.train(samples, recency_halflife_games=args.halflife)

    exp = model.export_json(fit, names)
    exp["trained_on_games"] = n_games
    exp["position_only"] = not args.with_skill
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(exp, fh, separators=(",", ":"))
    print(f"  wrote {args.out} ({os.path.getsize(args.out) // 1024} KB, "
          f"{len(names)} features, {len(exp['members'])} members)")

    if args.install:
        dst = os.path.join(mods_dir(), "coach-overlay")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(MOD_SRC, dst)
        print(f"  installed mod to {dst}")
        print("  Fully quit and relaunch 0 A.D. to pick it up.")


if __name__ == "__main__":
    main()
