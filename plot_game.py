#!/usr/bin/env python3
"""Render one game as a chess-engine-style eval bar with annotated swings.

Produces a PNG next to the replay directory (or wherever --out points).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib

matplotlib.use("Agg")               # no display backend needed
import matplotlib.pyplot as plt

from oadrep import coach, model, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", required=True, help="comma-separated aliases you play under")
    ap.add_argument("--opp", default="", help="comma-separated opponent aliases")
    ap.add_argument("--game", required=True, help="path to the replay directory to plot")
    ap.add_argument("--step", type=int, default=5,
                    help="prediction cadence in seconds (finer = smoother curve)")
    ap.add_argument("--halflife", type=int, default=200)
    ap.add_argument("--out", help="output PNG path (default: alongside the replay)")
    ap.add_argument("--top-swings", type=int, default=4)
    ap.add_argument("--root")
    args = ap.parse_args()

    root = schema.find_root(args.root)
    if not root:
        sys.exit("No replay directory found.")

    me = [n.strip() for n in args.me.split(",") if n.strip()]
    opps = {n.strip() for n in args.opp.split(",") if n.strip()} or None

    print("Loading games...")
    all_games = list(schema.iter_games(root))

    # Train on everything EXCEPT the game we're plotting, so the curve is
    # honest out-of-sample rather than a training artifact.
    target_dir = os.path.abspath(args.game).rstrip("/")
    train_games = [g for g in all_games
                   if os.path.abspath(g.directory).rstrip("/") != target_dir]

    print("Building training samples...")
    train_samples = model.build_samples(train_games, me, opp_aliases=opps,
                                        step=args.step, load_commands=True)
    print(f"  {len(train_samples)} samples from {len({s.game_dir for s in train_samples})} games")

    print("Building target-game samples...")
    target_game = next((g for g in all_games
                        if os.path.abspath(g.directory).rstrip("/") == target_dir), None)
    if target_game is None:
        sys.exit(f"Target game not found: {args.game}")
    target_samples = model.build_samples([target_game], me, opp_aliases=opps,
                                         step=args.step, load_commands=True)
    if not target_samples:
        sys.exit("Target game produced no samples (unfinished? wrong player?).")

    print("Training...")
    fit, names = model.train(train_samples,
                             recency_halflife_games=args.halflife or None)

    # All samples visible to the coach: train + target so we can slice by game_dir.
    all_samples = train_samples + target_samples
    swings = coach.swings_in_game(fit, all_samples, names,
                                  target_samples[0].game_dir,
                                  top_n=args.top_swings, min_drop=0.05)

    # Compute the curve for plotting.
    import numpy as np
    ts = np.array([s.t / 60.0 for s in sorted(target_samples, key=lambda s: s.t)])
    xs = np.array([[s.features.get(n, 0.0) for n in names]
                   for s in sorted(target_samples, key=lambda s: s.t)])
    p = fit.predict_proba(xs)[:, 1]
    outcome = "WON" if target_samples[0].label == 1 else "LOST"
    _, opp = target_game.perspective(me)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    # Shade above/below 50% so the eval bar reads at a glance.
    ax.fill_between(ts, 0.5, p, where=p >= 0.5, alpha=0.35,
                    color="#2c7fb8", label=f"{args.me.split(',')[0]} favoured")
    ax.fill_between(ts, 0.5, p, where=p < 0.5, alpha=0.35,
                    color="#e34a33", label=f"{opp.nick} favoured")
    ax.plot(ts, p, color="black", linewidth=1.4)
    ax.axhline(0.5, color="#555", linewidth=0.8, linestyle="--")

    # Annotate the top swings.
    for i, sw in enumerate(swings, start=1):
        t_mid = (sw.t_before + sw.t_after) / 120.0
        ax.annotate(f"#{i}: -{sw.drop:.0%}",
                    xy=(t_mid, sw.p_after),
                    xytext=(t_mid, sw.p_after - 0.18),
                    ha="center", fontsize=9, color="#b30000",
                    arrowprops=dict(arrowstyle="->", color="#b30000", lw=0.8))

    ax.set_ylim(0, 1)
    ax.set_xlim(0, ts[-1])
    ax.set_ylabel("P(win)")
    ax.set_xlabel("minutes")
    title_civs = f"{args.me.split(',')[0]} ({target_game.perspective(me)[0].civ})" \
                 f" vs {opp.nick} ({opp.civ})"
    ax.set_title(f"{title_civs}   —   {outcome} in {ts[-1]:.1f} min   "
                 f"({os.path.basename(target_game.directory)})")
    ax.legend(loc="lower left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out = args.out or os.path.join(target_game.directory, "eval_bar.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\nWrote {out}")

    # Also print swing legend so the labels have context.
    if swings:
        print("\nTop swings (labels on chart):")
        for i, sw in enumerate(swings, start=1):
            print(f"  #{i}  t={sw.t_before/60:.1f}min -> {sw.t_after/60:.1f}min   "
                  f"P {sw.p_before:.0%} -> {sw.p_after:.0%}   (drop {sw.drop:.0%})")
            for row in sw.top_deltas[:3]:
                if len(row) == 3:
                    name, contrib, delta = row
                    print(f"       {name:<34} logit {contrib:+.2f}  (feature {delta:+,.1f})")


if __name__ == "__main__":
    main()
