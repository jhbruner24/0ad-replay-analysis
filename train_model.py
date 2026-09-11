#!/usr/bin/env python3
"""Train and evaluate the win-probability model on the real collection."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oadrep import model, schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--me", required=True, help="comma-separated aliases you play under")
    ap.add_argument("--opp", default="", help="comma-separated opponent-nick aliases")
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--demo-game", help="draw a win-probability curve for this game directory")
    ap.add_argument("--root")
    args = ap.parse_args()

    root = schema.find_root(args.root)
    if not root:
        sys.exit("No replay directory found.")

    me = [n.strip() for n in args.me.split(",") if n.strip()]
    opps = {n.strip() for n in args.opp.split(",") if n.strip()} or None

    print("Loading games...")
    games = list(schema.iter_games(root))
    print(f"  {len(games)} directories total")

    samples = model.build_samples(games, me, opp_aliases=opps)
    if not samples:
        sys.exit("No usable samples. Check --me and --opp.")
    print(f"  {len({s.game_dir for s in samples})} decided 1v1s produced {len(samples)} samples")

    train_s, test_s, cut = model.temporal_split(samples, holdout_frac=args.holdout)
    print(f"\nTemporal split at order {cut}: {len(train_s)} train / {len(test_s)} test")

    print("\nTraining logistic regression with isotonic calibration...")
    fit, names = model.train(train_s, clip_quantile=model.DEFAULT_CLIP)
    print(f"  {len(names)} features")

    def _report(label, samples):
        m = model.evaluate(fit, samples, names)
        print(f"\n  {label:<6} n={m['n']:<5}  base rate {m['base_rate']:.1%}   "
              f"Brier {m['brier']:.3f}   log loss {m['logloss']:.3f}   AUC {m['auc']:.3f}")
        return m

    train_m = _report("train", train_s)
    test_m = _report("test", test_s)

    baseline = max(test_m["base_rate"], 1 - test_m["base_rate"])
    baseline_brier = test_m["base_rate"] * (1 - test_m["base_rate"])
    print(f"\n  Baseline Brier (always-predict-base-rate): {baseline_brier:.3f}")
    print(f"  Test Brier improvement over baseline: "
          f"{(baseline_brier - test_m['brier']) / baseline_brier:+.1%}")

    print("\n=== Reliability on test set ===")
    print(model.reliability_report(test_m["labels"], test_m["predictions"]))

    if args.demo_game:
        print(f"\n=== Win-probability curve for {os.path.basename(args.demo_game)} ===")
        ts, ps, label = model.curve_for_game(fit, samples, names, args.demo_game)
        if ts is None:
            print("  Game not in sample set.")
        else:
            outcome = "WON" if label == 1 else "LOST"
            print(f"  Result: {outcome}")
            for t, p in zip(ts, ps):
                bar = "#" * round(p * 40)
                mark = "*" if abs(p - 0.5) < 0.1 else " "
                print(f"  t={t/60:>5.1f}min  P(win)={p:>5.1%}  |{bar:<40}|{mark}")


if __name__ == "__main__":
    main()
