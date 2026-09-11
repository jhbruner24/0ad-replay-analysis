#!/usr/bin/env python3
"""Read the live_state.json the coach-overlay mod publishes, score it, print.

Terminal companion / debugging aid. The mod scores the game itself in-game
(export_model.py ships it the fitted numbers); this process independently
tails the state file, extracts the same features, and prints P(win) with a
text bar. Useful for checking the in-game number or trying --with-skill.

  python3 live_coach.py --me wace8000 --opp wolfmapking,noob5layer

Model is fit once at startup on all historical replays. Live scoring is
sub-millisecond after that.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oadrep import model, schema


# The mod writes live_state.json here.
MOD_DIR = "saves/campaigns/coach-overlay"


def user_data_dir() -> str:
    """Where the engine puts its user data (the VFS root for saves/)."""
    home = os.path.expanduser("~")
    if platform.system() == "Darwin":
        return f"{home}/Library/Application Support/0ad"
    if platform.system() == "Windows":
        return f"{home}/AppData/Roaming/0ad"
    return f"{home}/.local/share/0ad"


def state_file_path() -> str:
    return f"{user_data_dir()}/{MOD_DIR}/live_state.json"



def _bar(p: float, width: int = 40) -> str:
    filled = round(p * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _pick_perspective(state: dict, me_nicks: set[str], opp_nicks: set[str]):
    """Return (me_idx, them_idx, rating_diff) or (None, None, 0)."""
    players = state.get("players") or []
    # players[0] here is Gaia only if the mod copies playerStates literally;
    # v0.28 GetExtendedSimulationState returns players indexed by player_id
    # with 0 = Gaia, so we scan for our nick.
    def _split(name):
        import re
        m = re.match(r"^(\S+) \((\d+)\)$", name or "")
        return (m.group(1), int(m.group(2))) if m else (name, None)

    me_idx = them_idx = None
    me_rating = them_rating = None
    others = []
    for i, p in enumerate(players):
        if not isinstance(p, dict) or i == 0:  # 0 is Gaia
            continue
        nick, rating = _split(str(p.get("name", "")))
        if nick in me_nicks and me_idx is None:
            me_idx, me_rating = i, rating
        elif nick in opp_nicks and them_idx is None:
            them_idx, them_rating = i, rating
        else:
            others.append((i, rating))
    # In a 1v1 the opponent is whoever else is on the map, named or not.
    if me_idx is not None and them_idx is None and len(others) == 1:
        them_idx, them_rating = others[0]
    if me_idx is None or them_idx is None:
        return None, None, 0
    rating_diff = ((me_rating or 0) - (them_rating or 0)) if me_rating and them_rating else 0
    return me_idx, them_idx, rating_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", required=True, help="comma-separated aliases")
    ap.add_argument("--opp", default="", help="comma-separated opponent aliases")
    ap.add_argument("--step", type=int, default=15,
                    help="training cadence (matches historical-model build)")
    ap.add_argument("--halflife", type=int, default=200)
    ap.add_argument("--state-file", default=None,
                    help="override the live-state path (default: platform standard)")
    ap.add_argument("--root", help="replay root for training")
    ap.add_argument("--poll", type=float, default=0.5, help="file poll interval")
    ap.add_argument("--clip", type=float, default=model.DEFAULT_CLIP,
                    help="winsorise features at this training quantile")
    ap.add_argument("--with-skill", action="store_true",
                    help="keep the matchup prior and rating term (default: "
                         "position-only - mirrored training, 50%% at t=0)")
    args = ap.parse_args()

    root = schema.find_root(args.root)
    if not root:
        sys.exit("No replay directory found. Pass --root explicitly.")

    me_nicks = {n.strip() for n in args.me.split(",") if n.strip()}
    opp_nicks = {n.strip() for n in args.opp.split(",") if n.strip()}

    print("Loading historical replays...", flush=True)
    games = list(schema.iter_games(root))
    samples = model.build_samples(
        games, list(me_nicks), opp_aliases=opp_nicks or None,
        step=args.step, load_commands=True,
    )
    if not samples:
        sys.exit("No training samples produced from the historical corpus.")
    if not args.with_skill:
        samples = model.position_only(samples)
    print(f"  fitting on {len({s.game_dir for s in samples})} games "
          f"({len(samples)} samples"
          f"{', position-only' if not args.with_skill else ''})...", flush=True)
    fit, feature_names = model.train(samples, recency_halflife_games=args.halflife,
                                     clip_quantile=args.clip)
    # Command features live in feature_names too; we zero them out live because
    # the mod doesn't publish the command log. Record their default values.
    cmd_feature_defaults = {n: 0.0 for n in feature_names
                            if n.startswith("cmd.") or n.startswith("cat.")}
    print(f"  {len(feature_names)} features "
          f"({len(cmd_feature_defaults)} command features zeroed live).",
          flush=True)

    state_path = args.state_file or state_file_path()
    print(f"\nWatching {state_path}\n(Start a game; hit ^C to stop.)\n", flush=True)

    import numpy as np
    last_seen = 0
    last_reported_tick = -1
    while True:
        try:
            mtime = os.path.getmtime(state_path)
        except OSError:
            time.sleep(args.poll)
            continue
        if mtime == last_seen:
            time.sleep(args.poll)
            continue
        last_seen = mtime

        try:
            with open(state_path, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError):
            continue
        if state.get("tick") == last_reported_tick:
            continue
        last_reported_tick = state.get("tick", last_reported_tick)

        me_idx, them_idx, rating_diff = _pick_perspective(state, me_nicks, opp_nicks)
        if me_idx is None:
            names = [p.get("name") for p in (state.get("players") or [])
                     if isinstance(p, dict)]
            print(f"  (waiting for perspective; live players: {names})", flush=True)
            continue

        players = state["players"]
        feats = model.live_features(model.state_feature_spec(), players[me_idx], players[them_idx])
        feats["rating_diff"] = float(rating_diff)
        feats["rated"] = 1 if rating_diff else 0
        feats.update(cmd_feature_defaults)
        X = np.array([[feats.get(n, 0.0) for n in feature_names]])
        p = float(fit.predict_proba(X)[0, 1])
        t = state.get("t_seconds") or 0.0
        term_w = shutil.get_terminal_size(fallback=(80, 20)).columns
        line = f"t={t/60:>5.1f}min  P(win)={p:>5.1%}  {_bar(p, min(50, term_w-30))}"
        print("\r" + line, end="\n", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
