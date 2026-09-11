#!/usr/bin/env python3
"""End-to-end coach report: fit the model, find swings and blown leads."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oadrep import coach, model, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", required=True, help="comma-separated aliases you play under")
    ap.add_argument("--opp", default="", help="comma-separated opponent aliases")
    ap.add_argument("--step", type=int, default=15,
                    help="prediction cadence in seconds (finer=more precise swings)")
    ap.add_argument("--halflife", type=int, default=200,
                    help="recency halflife in games; 0 to disable")
    ap.add_argument("--holdout", type=float, default=0.20)
    ap.add_argument("--top-swings-per-game", type=int, default=3)
    ap.add_argument("--blown-lead-threshold", type=float, default=0.85)
    ap.add_argument("--worst-game", action="store_true",
                    help="print full swing report for the biggest blown lead")
    ap.add_argument("--root")
    args = ap.parse_args()

    root = schema.find_root(args.root)
    if not root:
        sys.exit("No replay directory found.")

    me = [n.strip() for n in args.me.split(",") if n.strip()]
    opps = {n.strip() for n in args.opp.split(",") if n.strip()} or None

    print("Loading games...")
    games = list(schema.iter_games(root))
    print(f"  {len(games)} directories")

    samples = model.build_samples(games, me, opp_aliases=opps,
                                  step=args.step, load_commands=True)
    if not samples:
        sys.exit("No samples produced.")
    print(f"  {len({s.game_dir for s in samples})} decided 1v1s -> {len(samples)} samples")

    train_s, test_s, cut = model.temporal_split(samples, holdout_frac=args.holdout)
    hl = args.halflife or None
    print(f"\nTraining (recency halflife: {hl or 'none'})...")
    fit, names = model.train(train_s, recency_halflife_games=hl)
    print(f"  {len(names)} features")

    r = model.evaluate(fit, test_s, names)
    base = r["base_rate"] * (1 - r["base_rate"])
    print(f"  test:  n={r['n']}  Brier {r['brier']:.4f}  AUC {r['auc']:.3f}  "
          f"vs baseline {base:.4f}  ({(base - r['brier']) / base:+.1%})")

    print("\n=== Blown leads (games you led >=%.0f%% and lost) ===" %
          (args.blown_lead_threshold * 100))
    leads = coach.blown_leads(fit, samples, names,
                              threshold=args.blown_lead_threshold)
    print(coach.format_blown_leads(leads[:10]))

    if args.worst_game and leads:
        worst = leads[0]
        print(f"\n=== Swings in {worst['game_dir'].split('/')[-1]} "
              f"(peak {worst['peak_p']:.0%}, lost) ===")
        for sw in coach.swings_in_game(fit, samples, names, worst["game_dir"]):
            print(coach.format_swing(sw))


if __name__ == "__main__":
    main()
