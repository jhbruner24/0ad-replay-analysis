"""Compare your stat trajectories in games you won against games you lost.

This is deliberately not a model. With a few hundred games it is better powered
than anything learned, it needs no training or validation split, and its output is
directly actionable: "your food gathering flattens between 6 and 9 minutes in the
games you lose."

Method, per statistic and per time bucket:

  - split your own series into won-games and lost-games
  - compare the two distributions with a rank-based effect size (the probability
    that a random winning game exceeds a random losing one), which needs no
    normality assumption and is not moved by a couple of blowouts
  - attach a bootstrap confidence interval, and report only buckets whose
    interval excludes chance

Effect size is the common-language statistic: 0.5 is no difference, 0.8 means a
randomly chosen won game beats a randomly chosen lost game 80% of the time.

Multiple comparisons are real here: dozens of stats times several buckets means
some intervals exclude 0.5 by luck. Treat this as a ranked list of leads to
investigate, not a set of confirmed findings.

    python3 -m oadrep.compare --me "YourName"
"""

from __future__ import annotations

import argparse
import random
import statistics
from collections import defaultdict

from . import schema

DEFAULT_BUCKETS = [(0, 300), (300, 600), (600, 900), (900, 1200), (1200, 1_000_000)]

# Statistics that directly record combat results. These separate wins from losses
# almost by definition - "you killed more and lost fewer in the games you won" is
# a restatement of winning, not advice. They are reported separately so they do not
# crowd out the stats that reflect decisions you could actually make differently.
OUTCOME_COUPLED = {
    "unitsLost", "unitsLostValue",
    "enemyUnitsKilled", "enemyUnitsKilledValue",
    "unitsCaptured", "unitsCapturedValue",
    "buildingsLost", "buildingsLostValue",
    "enemyBuildingsDestroyed", "enemyBuildingsDestroyedValue",
    "buildingsCaptured", "buildingsCapturedValue",
    "lootCollected",
}


def is_outcome_coupled(stat: str) -> bool:
    return stat.split(".")[0] in OUTCOME_COUPLED


def bucket_label(lo, hi):
    return f"{lo//60:>2}-{hi//60:<3}min" if hi < 1_000_000 else f"{lo//60:>2}+     min"


def value_in_bucket(times, values, lo, hi):
    """Last observed value inside the window. Counters are cumulative, so the
    final reading is the total accrued by the end of that window."""
    picked = [v for t, v in zip(times, values)
              if lo <= t < hi and isinstance(v, (int, float))]
    return picked[-1] if picked else None


def common_language_effect(a, b):
    """P(random draw from `a` > random draw from `b`), ties counted as half.

    Computed from the Mann-Whitney U statistic via midranks, which is
    O((n+m) log(n+m)) rather than the O(n*m) of comparing every pair - the
    difference matters because the bootstrap calls this thousands of times.
    """
    if not a or not b:
        return None
    n, m = len(a), len(b)
    merged = sorted([(v, 0) for v in a] + [(v, 1) for v in b])

    rank_sum_a = 0.0
    i = 0
    while i < len(merged):
        j = i
        while j + 1 < len(merged) and merged[j + 1][0] == merged[i][0]:
            j += 1
        midrank = (i + j) / 2.0 + 1.0          # ranks are 1-based
        rank_sum_a += midrank * sum(1 for k in range(i, j + 1) if merged[k][1] == 0)
        i = j + 1

    u = rank_sum_a - n * (n + 1) / 2.0
    return u / (n * m)


def bootstrap_ci(a, b, rounds=2000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    stats = []
    for _ in range(rounds):
        ra = [rng.choice(a) for _ in a]
        rb = [rng.choice(b) for _ in b]
        eff = common_language_effect(ra, rb)
        if eff is not None:
            stats.append(eff)
    if not stats:
        return None, None
    stats.sort()
    lo = stats[int(alpha / 2 * len(stats))]
    hi = stats[min(int((1 - alpha / 2) * len(stats)), len(stats) - 1)]
    return lo, hi


def collect(games, me, buckets=DEFAULT_BUCKETS, min_games=8):
    """-> {(stat, bucket): {"won": [...], "lost": [...]}} for decided 1v1s."""
    data = defaultdict(lambda: {"won": [], "lost": []})
    used = 0
    for game in games:
        if not game.decided:
            continue
        pair = game.perspective(me)
        if not pair:
            continue
        mine, _ = pair
        if mine.won is None or not mine.times:
            continue
        used += 1
        side = "won" if mine.won else "lost"
        for stat, series in mine.sequences.items():
            for lo, hi in buckets:
                val = value_in_bucket(mine.times, series, lo, hi)
                if val is not None:
                    data[(stat, (lo, hi))][side].append(val)
    return data, used


def analyse(data, min_per_side=6, seed=0):
    findings = []
    for (stat, bucket), sides in data.items():
        won, lost = sides["won"], sides["lost"]
        if len(won) < min_per_side or len(lost) < min_per_side:
            continue
        effect = common_language_effect(won, lost)
        lo, hi = bootstrap_ci(won, lost, seed=seed)
        if effect is None or lo is None:
            continue
        findings.append({
            "stat": stat, "bucket": bucket, "effect": effect, "ci": (lo, hi),
            "n_won": len(won), "n_lost": len(lost),
            "median_won": statistics.median(won),
            "median_lost": statistics.median(lost),
            "significant": lo > 0.5 or hi < 0.5,
        })
    findings.sort(key=lambda f: abs(f["effect"] - 0.5), reverse=True)
    return findings


def _table(rows, limit):
    print(f"  {'stat':<34} {'window':<12} {'effect':>7}  {'95% CI':<16} "
          f"{'median won':>12} {'median lost':>12}")
    print("  " + "-" * 100)
    for f in rows[:limit]:
        lo, hi = f["ci"]
        arrow = "higher" if f["effect"] > 0.5 else "LOWER "
        print(f"  {f['stat']:<34} {bucket_label(*f['bucket']):<12} "
              f"{f['effect']:>6.2f}  [{lo:.2f}, {hi:.2f}]   "
              f"{f['median_won']:>12,.0f} {f['median_lost']:>12,.0f}   {arrow} in wins")


def report(findings, used, me, limit=15, only_significant=True):
    print(f"\nWins-vs-losses comparison for {me}  ({used} decided 1v1 games)")
    if not findings:
        print("  Not enough data. Need at least a handful of games on each side.")
        return

    shown = [f for f in findings if f["significant"]] if only_significant else findings
    decisions = [f for f in shown if not is_outcome_coupled(f["stat"])]
    outcomes = [f for f in shown if is_outcome_coupled(f["stat"])]

    print("\n=== Decision statistics ===")
    print("  Economy, expansion and production - things you chose. Look here first.\n")
    if decisions:
        _table(decisions, limit)
    else:
        print("  Nothing separates wins from losses beyond chance.")
        print("  That is a real result: at this sample size your economic choices")
        print("  do not distinguish your wins. Either the deciding factors are not")
        print("  in these stats (composition, timing, engagement quality), or games")
        print("  turn on execution the summary statistics never see.")

    if outcomes:
        print("\n=== Outcome-coupled statistics ===")
        print("  Combat results. These separate wins from losses almost by")
        print("  definition, so they confirm the data is sane but coach nothing.\n")
        _table(outcomes, min(limit, 5))

    print("\n  effect = P(a random won game exceeds a random lost game).")
    print("  0.50 is no difference. Ranked by distance from 0.50.")
    print("\n  Caution: dozens of stat/window pairs are tested, so some intervals")
    print("  exclude 0.50 by chance. Treat these as leads, not conclusions --")
    print("  the ones worth trusting are those that make tactical sense.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", help="replay directory (auto-detected if omitted)")
    ap.add_argument("--me", required=True, help="your in-game player name")
    ap.add_argument("--limit", type=int, default=15)
    ap.add_argument("--all", action="store_true", help="include non-significant rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = schema.find_root(args.root)
    if not root:
        raise SystemExit("No replay directory found. Pass --root /path/to/replays")

    data, used = collect(list(schema.iter_games(root)), args.me)
    report(analyse(data, seed=args.seed), used, args.me,
           limit=args.limit, only_significant=not args.all)


if __name__ == "__main__":
    main()
