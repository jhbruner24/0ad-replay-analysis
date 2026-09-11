#!/usr/bin/env python3
"""Read the live_state.json the coach-overlay mod publishes, score it, print.

Pairs with the coach-overlay mod. The mod writes ~/Library/Application Support/
0ad/saves/campaigns/coach-overlay/live_state.json every N seconds during a live game; this process
tails the file, extracts features the model was trained on, and prints P(win)
with a small text bar.

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


# The mod writes live_state.json here and reads p_win.json back from the same
# directory to draw the in-game panel.
MOD_DIR = "saves/campaigns/coach-overlay"

# The engine's VFS records a file's size when it first sees it and reads exactly
# that many bytes on every subsequent load (lib/file/vfs/vfs.cpp, LoadFile). The
# mod creates p_win.json at this size and we must overwrite it at this size, or
# the game reads a truncated/stale document. Keep in sync with coach_overlay.js.
P_WIN_BYTES = 512


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


def p_win_path() -> str:
    return f"{user_data_dir()}/{MOD_DIR}/p_win.json"


def write_p_win(path: str, payload: dict) -> None:
    """Overwrite p_win.json at exactly P_WIN_BYTES (space-padded), atomically."""
    body = json.dumps(payload, separators=(",", ":"))
    if len(body) > P_WIN_BYTES:
        raise ValueError(f"p_win payload {len(body)}B exceeds {P_WIN_BYTES}B")
    body = body + " " * (P_WIN_BYTES - len(body))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as fh:
        fh.write(body)
    os.replace(tmp, path)


def _flatten_sequences(raw):
    """Same rule as schema.flatten_sequences, without importing Player parsing."""
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


def _live_features(state: dict, me_idx: int, them_idx: int, feature_names: list[str],
                   rating_diff: float, cmd_feature_defaults: dict) -> dict:
    """Turn a single live-state payload into a feature dict for the model.

    Command features are set to zero: the live pipeline does not (yet) see the
    command log, so we hand the model a stats-only view. Because command
    coefficients are small, the resulting P(win) is very close to the full
    model's output (see the +0.5% AUC delta we measured).
    """
    t = float(state.get("t_seconds") or 0.0)
    feats = {"t_log": math.log(t + 1),
             "rating_diff": float(rating_diff),
             "rated": 1 if rating_diff else 0}
    for k, v in cmd_feature_defaults.items():
        feats[k] = v

    players = state.get("players") or []
    if me_idx >= len(players) or them_idx >= len(players):
        return feats
    mine = players[me_idx]
    them = players[them_idx]
    mine_seq = _flatten_sequences(mine.get("sequences"))
    them_seq = _flatten_sequences(them.get("sequences"))
    times = (mine.get("sequences") or {}).get("time") or []
    # last recorded observation
    def _last(series):
        if not isinstance(series, list):
            return None
        for v in reversed(series):
            if isinstance(v, (int, float)):
                return float(v)
        return None
    for name in feature_names:
        if not name.startswith("diff."):
            continue
        stat = name[len("diff."):]
        my_v = _last(mine_seq.get(stat))
        their_v = _last(them_seq.get(stat))
        if my_v is None:
            continue
        feats[name] = my_v - float(their_v or 0)
    return feats


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
    fit, feature_names = model.train(samples, recency_halflife_games=args.halflife)
    # Command features live in feature_names too; we zero them out live because
    # the mod doesn't publish the command log. Record their default values.
    cmd_feature_defaults = {n: 0.0 for n in feature_names
                            if n.startswith("cmd.") or n.startswith("cat.")}
    print(f"  {len(feature_names)} features "
          f"({len(cmd_feature_defaults)} command features zeroed live).",
          flush=True)

    state_path = args.state_file or state_file_path()
    out_path = (os.path.join(os.path.dirname(args.state_file), "p_win.json")
                if args.state_file else p_win_path())
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

        feats = _live_features(state, me_idx, them_idx, feature_names,
                               rating_diff, cmd_feature_defaults)
        X = np.array([[feats.get(n, 0.0) for n in feature_names]])
        p = float(fit.predict_proba(X)[0, 1])
        t = state.get("t_seconds") or 0.0
        term_w = shutil.get_terminal_size(fallback=(80, 20)).columns
        line = f"t={t/60:>5.1f}min  P(win)={p:>5.1%}  {_bar(p, min(50, term_w-30))}"
        print("\r" + line, end="\n", flush=True)
        try:
            write_p_win(out_path, {"schema": 1, "p": round(p, 4),
                                   "t_seconds": t, "tick": state.get("tick"),
                                   "written_at_ms": int(time.time() * 1000)})
        except OSError as e:
            print(f"  (could not write {out_path}: {e})", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
