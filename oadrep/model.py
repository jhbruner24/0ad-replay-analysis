"""Win-probability model for 1v1 games of 0 A.D.

Predicts P(you win) as a function of game state at time t. The model is
called at each 30-second snapshot, so its output over game time is the
eval-bar curve that visualisation will render.

Design choices, all driven by the data (see 0ad-replay-coach/README.md for why):

- **Differential features, not per-player values.** In a 1v1 only me-minus-them
  matters, so features are the difference of the two players' stats. This
  imposes the antisymmetry the problem actually has and halves the feature
  count.
- **Logistic regression, not gradient boosting.** ~700 decided games gives ~700
  independent outcomes, not the 20,000 rows the naive count suggests. Boosting
  memorises which game a snapshot came from and reports a fantasy validation
  score. Logistic regression yields calibrated probabilities almost for free.
- **Temporal split.** The matchup drifts, so a random split leaks the future
  into the past. Train on the older games, test on the newer.
- **Time as a feature.** The base rate shifts with game length -- long games
  end differently from short ones, and both players' stats accumulate over
  time. The model sees log(t+1) so early minutes get resolution and late
  ones flatten.
- **Rating differential as a static feature.** The strongest single predictor.
- **No outcome-coupled features.** Excludes combat results (unitsLost,
  enemyUnitsKilled, etc.) that separate wins from losses by definition. See
  compare.OUTCOME_COUPLED.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import compare, schema

# Feature stats are matched by prefix against the already-flattened sequence
# keys (schema.flatten_sequences produces e.g. "resourcesGathered.food",
# "unitsTrained.Infantry"). Chosen because they are decisions rather than
# outcomes (see compare.OUTCOME_COUPLED for the excluded list).
FEATURE_PREFIXES = [
    "populationCount",
    "percentMapControlled",
    "percentMapExplored",
    "unitsTrained.",
    "buildingsConstructed.",
    "resourcesGathered.",
    "resourcesUsed.",
    "resourcesCount.",
]


# Position features: what is on the map *now*, not what has happened so far.
# Raw sequences are mostly cumulative counters (resourcesUsed, unitsTrained,
# ...) that never go down, so a model fit on them cannot see a lead evaporate:
# a player who spent 20k metal on an army keeps the credit after the army is
# dead. These derive current state instead. Each entry is
#   name: {"terms": [(stat, coef), ...], "window": W or None, "live": key or None}
#   value(t) = sum(coef * stat(t)) over terms; with a window W it is a rate,
#   (S(t) - S(t-W)) / W.
# Every feature enters the model as a differential, mine - theirs. The
# exported model.json carries this spec and coach_overlay.js evaluates it the
# same way.
#
# "live": in a running game GetExtendedSimulationState carries exact current
# counts (popCount, resourceCounts, classCounts) that replays do not keep, and
# which are also better than the derived value: the engine does not count
# units their owner deleted as lost (Health.Kill skips KilledBy), and the
# `total` bucket of unitsLost/buildingsLost is never incremented at all, so
# trained - lost drifts high. The live key names the exact source; the
# derived terms are what training uses.
UNIT_CLASSES = ("Infantry", "Cavalry", "Champion", "Siege", "Ship", "Hero",
                "Worker", "Trader")
# Rough template cost per unit class, for valuing an army from counts. The
# classes do not overlap (citizen infantry are Workers; champions and heroes
# are not), so summing over them counts each unit once.
UNIT_COST = {"Worker": 110, "Cavalry": 180, "Champion": 260, "Siege": 380,
             "Ship": 220, "Hero": 600, "Trader": 180}
BUILDING_CLASSES = ("Structure", "CivCentre", "Fortress", "House", "Military",
                    "Economic", "Outpost", "Wonder")
RESOURCES = ("food", "wood", "stone", "metal")
RATE_WINDOW = 60.0
# Winsorisation quantile for train(); 0.15 maximised holdout AUC (0.782 vs
# 0.742 unclipped) on the wace8000 corpus and bounds extrapolation.
DEFAULT_CLIP = 0.15


def _entry(terms, window=None, live=None):
    return {"terms": terms, "window": window, "live": live}


def state_feature_spec(material=True):
    spec = {
        "state.pop": _entry([("populationCount", 1)], live="popCount"),
        "state.map_control": _entry([("percentMapControlled", 1)]),
        "state.map_explored": _entry([("percentMapExplored", 1)]),
        "income.trade": _entry([("tradeIncome", 1)], RATE_WINDOW),
        "momentum.value_lost": _entry([("unitsLostValue", 1), ("buildingsLostValue", 1)], RATE_WINDOW),
    }
    if material:
        # Value of everything still standing: all resources ever spent, less
        # the value of what has been lost. Techs count as standing. Split into
        # the army (units alive x rough template cost, so the live exact
        # counts apply) and the rest (buildings + techs), because a big tech
        # bill with no army is a very different position from the reverse.
        army = [(f"unitsTrained.{c}", cost) for c, cost in UNIT_COST.items()] \
             + [(f"unitsLost.{c}", -cost) for c, cost in UNIT_COST.items()] \
             + [(f"unitsCaptured.{c}", cost) for c, cost in UNIT_COST.items()]
        spec["material.army"] = _entry(army, live="army_value")
        spec["material.infra"] = _entry([(f"resourcesUsed.{r}", 1) for r in RESOURCES]
                                        + [("unitsLostValue", -1), ("buildingsLostValue", -1)]
                                        + [(st, -c) for st, c in army])
    for r in RESOURCES:
        spec[f"state.res.{r}"] = _entry([(f"resourcesCount.{r}", 1)], live=f"resourceCounts.{r}")
        spec[f"income.{r}"] = _entry([(f"resourcesGathered.{r}", 1)], RATE_WINDOW)
    for c in UNIT_CLASSES:
        spec[f"alive.units.{c}"] = _entry([(f"unitsTrained.{c}", 1), (f"unitsLost.{c}", -1),
                                           (f"unitsCaptured.{c}", 1)], live=f"classCounts.{c}")
    for c in BUILDING_CLASSES:
        spec[f"alive.buildings.{c}"] = _entry([(f"buildingsConstructed.{c}", 1),
                                               (f"buildingsLost.{c}", -1),
                                               (f"buildingsCaptured.{c}", 1)],
                                              live=f"classCounts.{c}")
    return spec


def _live_lookup_path(player_state, key):
    """`a.b` -> player_state["a"]["b"]; a missing classCounts key means 0."""
    cur = player_state
    for part in key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    if cur is None:
        return 0.0 if key.startswith("classCounts.") else None
    return float(cur) if isinstance(cur, (int, float)) else None


def _live_lookup(player_state, key):
    if key == "army_value":
        counts = player_state.get("classCounts")
        if not isinstance(counts, dict):
            return None
        return float(sum(cost * (counts.get(c) or 0) for c, cost in UNIT_COST.items()))
    return _live_lookup_path(player_state, key)


def spec_value(times, seqs, terms, window, t):
    """Evaluate one spec entry for one player at time t. None if no data."""
    def total_at(tt):
        total, seen = 0.0, False
        for stat, coef in terms:
            series = seqs.get(stat)
            if not series:
                continue
            v = _value_at(times, series, tt)
            if v is None:
                continue
            seen = True
            total += coef * v
        return total if seen else None
    now = total_at(t)
    if now is None:
        return None
    if window is None:
        return now
    before = total_at(t - window)
    return (now - (0.0 if before is None else before)) / window


def diff_features(spec, t, mine_times, mine_seqs, them_times, them_seqs,
                  mine_live=None, them_live=None):
    """The perspective-free part of a feature dict: t_log plus one
    `diff.<name>` per spec entry, mine - theirs (theirs 0 if absent).

    `mine_live`/`them_live` are the per-player objects from a running game's
    GetExtendedSimulationState; when given, entries with a "live" key read
    the exact current value from there instead of deriving it.
    """
    feats = {"t_log": math.log(t + 1)}
    for name, e in spec.items():
        my_v = their_v = None
        if e.get("live") and mine_live is not None and them_live is not None:
            my_v = _live_lookup(mine_live, e["live"])
            their_v = _live_lookup(them_live, e["live"])
        if my_v is None:
            my_v = spec_value(mine_times, mine_seqs, e["terms"], e["window"], t)
            their_v = spec_value(them_times, them_seqs, e["terms"], e["window"], t)
        if my_v is None:
            continue
        feats[f"diff.{name}"] = my_v - (their_v or 0.0)
    return feats


def live_features(spec, mine_live, them_live):
    """Feature dict for one live GetExtendedSimulationState player pair.
    Uses the last sequence entry as "now" for derived stats; exact live counts
    for entries that have them."""
    mine_seq = flatten_sequences(mine_live.get("sequences"))
    them_seq = flatten_sequences(them_live.get("sequences"))
    mine_times = (mine_live.get("sequences") or {}).get("time") or []
    them_times = (them_live.get("sequences") or {}).get("time") or []
    t = float(mine_times[-1]) if mine_times else 0.0
    return diff_features(spec, t, mine_times, mine_seq, them_times, them_seq,
                         mine_live, them_live)


def flatten_sequences(raw):
    """`{"a": [..], "b": {"x": [..]}}` -> `{"a": [..], "b.x": [..]}`; drops time."""
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


def _value_at(times, values, t):
    """Value at or just before time t. Times are ascending; values counter-like."""
    i = min(bisect.bisect_right(times, t), len(values)) - 1
    while i >= 0:
        vi = values[i]
        if isinstance(vi, (int, float)):
            return vi
        i -= 1
    return None


def _select_features(sequences):
    """Which of the flattened keys we treat as model features."""
    return [k for k in sequences
            if any(k == p or k.startswith(p) for p in FEATURE_PREFIXES)
            and not compare_is_outcome_coupled(k)]


def compare_is_outcome_coupled(stat):
    """Local re-import to keep this module usable standalone."""
    from . import compare
    return compare.is_outcome_coupled(stat)


@dataclass
class Sample:
    game_dir: str
    order: int
    t: float
    features: dict
    label: int          # 1 if `me` won
    weight: float = 1.0


def build_samples(games, me, opp_aliases=None, ts=None, step=30, until=1800,
                  cmd_window=30, load_commands=True, spec="state"):
    """One sample per (game, time-of-observation).

    `spec`: "state" (default) derives position features via
    `state_feature_spec`; None uses the raw cumulative sequence stats, kept for
    comparison.

    By default emits a sample every `step` seconds up to `until` seconds. The
    stat features come from the 30-second `sequences` (last-observation-carried-
    forward, so successive samples inside one 30s window share stats). The
    command features come from the last `cmd_window` seconds and DO refresh
    every step, so a `step=5` grid delivers real 5-second-resolution updates
    even though the stats sampler is fixed at 30 seconds.

    Set `load_commands=False` to skip parsing commands.txt (much faster,
    stats-only model - use it as a baseline).
    """
    from . import commands as commands_mod
    if spec == "state":
        spec = state_feature_spec()
    if ts is None:
        ts = tuple(range(step, until + 1, step))
    samples = []
    for game in games:
        if not game.decided:
            continue
        pair = game.perspective(me)
        if not pair:
            continue
        mine, them = pair
        if opp_aliases and them.nick not in opp_aliases:
            continue
        # Rating differential is only defined when both are rated; use 0 as an
        # honest fallback so unrated games still contribute, with a flag.
        if mine.rating is not None and them.rating is not None:
            rating_diff = mine.rating - them.rating
            rated = 1
        else:
            rating_diff = 0
            rated = 0

        feat_names = _select_features(mine.sequences)
        if not feat_names:
            continue
        mine_times, them_times = mine.times, them.times

        events = commands_mod.parse_for_game(game.directory) if load_commands else []
        game_len = mine.times[-1] if mine.times else 0
        for t in ts:
            if not mine.times or t > game_len:
                continue
            if spec:
                feats = diff_features(spec, t, mine_times, mine.sequences,
                                      them_times, them.sequences)
            else:
                feats = {"t_log": math.log(t + 1)}
            feats["rating_diff"] = rating_diff
            feats["rated"] = rated
            if load_commands:
                mine_cmd = commands_mod.counts_in_window(events, mine.player_id,
                                                        t - cmd_window, t)
                their_cmd = commands_mod.counts_in_window(events, them.player_id,
                                                         t - cmd_window, t)
                mine_total = sum(mine_cmd.values())
                their_total = sum(their_cmd.values())
                feats["cmd.total_diff"] = float(mine_total - their_total)
                feats["cmd.total_mine"] = float(mine_total)
                # Per-type differentials, unchanged.
                for ctype in commands_mod.TRACKED_TYPES + ("_other",):
                    feats[f"cmd.{ctype}_diff"] = float(
                        mine_cmd.get(ctype, 0) - their_cmd.get(ctype, 0))
                # Category rollups: what mode is each player in.
                mine_cat = commands_mod.category_counts(mine_cmd)
                their_cat = commands_mod.category_counts(their_cmd)
                for cat in commands_mod.CATEGORIES:
                    feats[f"cat.{cat}_diff"] = float(mine_cat[cat] - their_cat[cat])
                # Diversity of action mix.
                feats["cmd.entropy_diff"] = (commands_mod.shannon_entropy(mine_cmd)
                                             - commands_mod.shannon_entropy(their_cmd))
                # Burst signal: are things happening faster than usual?
                feats["cmd.accel_mine"] = commands_mod.rate_acceleration(
                    events, mine.player_id, t)
                feats["cmd.accel_their"] = commands_mod.rate_acceleration(
                    events, them.player_id, t)
            if not spec:
                for name in feat_names:
                    my_series = mine.sequences.get(name)
                    their_series = them.sequences.get(name)
                    my_v = _value_at(mine.times, my_series, t) if my_series else None
                    their_v = _value_at(them.times, their_series, t) if their_series else 0
                    if my_v is None:
                        continue
                    feats[f"diff.{name}"] = float(my_v) - float(their_v or 0)
            samples.append(Sample(
                game_dir=game.directory, order=game.order, t=t,
                features=feats, label=int(mine.won),
            ))
    return samples


def position_only(samples):
    """Strip who-is-playing from a sample set: mirror every sample and drop skill.

    Each sample is duplicated from the opponent's perspective (differentials
    negated, label flipped) and the rating features are zeroed. The fitted
    model then has no intercept to lean on and no rating term: at t=0 it says
    50%, and afterwards it only expresses which side of the differential it
    would rather be on. The one-sided command features (`cmd.total_mine`)
    have no mirror and are dropped; the accel pair is swapped.

    The corpus is still one matchup, so *what a lead is worth* is learned from
    how these two players resolved it - but not *who* holds it.
    """
    out = []
    for s in samples:
        a, b = {}, {}
        for k, v in s.features.items():
            if k in ("rating_diff", "rated"):
                a[k] = b[k] = 0.0
            elif k == "cmd.total_mine":
                continue
            elif k == "cmd.accel_mine":
                a[k] = v
                b["cmd.accel_their"] = v
            elif k == "cmd.accel_their":
                a[k] = v
                b["cmd.accel_mine"] = v
            elif k.startswith("diff.") or k.endswith("_diff"):
                a[k] = v
                b[k] = -v
            else:  # t_log and anything else perspective-free
                a[k] = b[k] = v
        out.append(Sample(game_dir=s.game_dir, order=s.order, t=s.t,
                          features=a, label=s.label))
        out.append(Sample(game_dir=s.game_dir, order=s.order, t=s.t,
                          features=b, label=1 - s.label))
    return out


def _to_matrix(samples):
    names = sorted({k for s in samples for k in s.features.keys()})
    X = np.array([[s.features.get(n, 0.0) for n in names] for s in samples])
    y = np.array([s.label for s in samples])
    groups = np.array([s.order for s in samples])
    return X, y, groups, names


def temporal_split(samples, holdout_frac=0.25):
    """Split by game order, not by row. Newest `holdout_frac` of games -> test."""
    orders = sorted({s.order for s in samples})
    cut = orders[int(len(orders) * (1 - holdout_frac))]
    train = [s for s in samples if s.order < cut]
    test = [s for s in samples if s.order >= cut]
    return train, test, cut


def train(samples, recency_halflife_games=None, clip_quantile=None):
    """Fit a calibrated logistic model on a set of samples.

    `clip_quantile`: if set (e.g. 0.02), winsorise every feature to its
    [q, 1-q] training quantiles before standardising, at fit and predict
    time. A linear model extrapolates without limit, so one feature at a
    value never seen in training (a 40k infrastructure lead) can outvote every
    other feature; clipping bounds any single feature's say to what the data
    actually supported.

    `recency_halflife_games`: if set, weight each sample by 0.5 ** (games_ago /
    halflife) where games_ago is the rank of that game's order within the
    training set counting back from newest. A halflife of 100 means a game 200
    games older has 1/4 the weight of the most recent game. This fights the
    matchup drift the census showed. None disables weighting.
    """
    X, y, _, names = _to_matrix(samples)
    lo = hi = None
    if clip_quantile:
        lo = np.quantile(X, clip_quantile, axis=0)
        hi = np.quantile(X, 1 - clip_quantile, axis=0)
        X = np.clip(X, lo, hi)
    # Standardise manually. Wrapping LR in a Pipeline breaks sample_weight
    # forwarding through CalibratedClassifierCV (sklearn issue #21134).
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    weights = None
    if recency_halflife_games:
        orders = sorted({s.order for s in samples})
        rank = {o: i for i, o in enumerate(reversed(orders))}   # 0 = newest game
        weights = np.array([0.5 ** (rank[s.order] / recency_halflife_games)
                            for s in samples])

    base = LogisticRegression(C=0.5, max_iter=1000)
    fitted = CalibratedClassifierCV(base, method="isotonic", cv=5)
    fit_kwargs = {"sample_weight": weights} if weights is not None else {}
    fitted.fit(Xz, y, **fit_kwargs)
    return _StandardisedModel(fitted, mu, sd, lo, hi), names


@dataclass
class _StandardisedModel:
    """Bundles the standardisation with the fitted model for prediction."""
    inner: object
    mu: np.ndarray
    sd: np.ndarray
    lo: np.ndarray | None = None
    hi: np.ndarray | None = None

    def predict_proba(self, X):
        if self.lo is not None:
            X = np.clip(X, self.lo, self.hi)
        return self.inner.predict_proba((X - self.mu) / self.sd)


def export_json(fit, names, spec=None):
    """Serialise a trained model so something without sklearn can score it.

    Consumed by the coach-overlay mod's JS (gui/session/coach_overlay.js),
    which must produce the same number sklearn does. The calibrated model is
    an average over `cv` members, each a logistic regression followed by an
    isotonic map applied to its decision function (linear interpolation
    between thresholds, clipped at the ends). `predict_from_export` below is
    the reference implementation; the JS mirrors it line for line.
    """
    members = []
    for cc in fit.inner.calibrated_classifiers_:
        lr = cc.estimator
        iso = cc.calibrators[0]
        members.append({
            "coef": [float(c) for c in lr.coef_[0]],
            "intercept": float(lr.intercept_[0]),
            "iso_x": [float(x) for x in iso.X_thresholds_],
            "iso_y": [float(y) for y in iso.y_thresholds_],
        })
    return {
        "schema": 2,
        "spec": {name: {"terms": [[st, c] for st, c in e["terms"]],
                        "window": e["window"], "live": e["live"]}
                 for name, e in (spec or {}).items()},
        "names": list(names),
        "mu": [float(v) for v in fit.mu],
        "sd": [float(v) for v in fit.sd],
        "lo": None if fit.lo is None else [float(v) for v in fit.lo],
        "hi": None if fit.hi is None else [float(v) for v in fit.hi],
        "members": members,
    }


def predict_from_export(exp, feats):
    """Pure-Python scoring of an `export_json` dict. Reference for the JS."""
    z = []
    for i, n in enumerate(exp["names"]):
        v = feats.get(n, 0.0)
        if exp.get("lo") is not None:
            v = min(max(v, exp["lo"][i]), exp["hi"][i])
        z.append((v - exp["mu"][i]) / exp["sd"][i])
    total = 0.0
    for m in exp["members"]:
        d = m["intercept"] + sum(c * v for c, v in zip(m["coef"], z))
        xs, ys = m["iso_x"], m["iso_y"]
        if d <= xs[0]:
            p = ys[0]
        elif d >= xs[-1]:
            p = ys[-1]
        else:
            i = bisect.bisect_right(xs, d) - 1
            x0, x1, y0, y1 = xs[i], xs[i + 1], ys[i], ys[i + 1]
            p = y0 if x1 == x0 else y0 + (y1 - y0) * (d - x0) / (x1 - x0)
        total += p
    return total / len(exp["members"])


def evaluate(model, samples, names):
    X = np.array([[s.features.get(n, 0.0) for n in names] for s in samples])
    y = np.array([s.label for s in samples])
    p = model.predict_proba(X)[:, 1]
    return {
        "n": len(y),
        "base_rate": float(y.mean()),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p, labels=[0, 1]),
        "auc": roc_auc_score(y, p) if len(set(y)) > 1 else float("nan"),
        "predictions": p,
        "labels": y,
    }


def reliability_report(y_true, p_pred, bins=10):
    """Text-mode calibration curve. A well-calibrated model has predicted ~ observed."""
    lines = ["  bucket        n     predicted  observed"]
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_pred >= lo) & (p_pred < hi if hi < 1.0 else p_pred <= hi)
        if mask.sum() == 0:
            continue
        lines.append(f"  {lo:.2f}-{hi:.2f}  {mask.sum():>5}     "
                     f"{p_pred[mask].mean():>6.1%}    {y_true[mask].mean():>6.1%}")
    return "\n".join(lines)


def curve_for_game(model, samples, names, game_dir):
    """Every prediction for one game, sorted by time. For the eval-bar demo."""
    hits = [s for s in samples if s.game_dir == game_dir]
    hits.sort(key=lambda s: s.t)
    if not hits:
        return None, None, None
    X = np.array([[s.features.get(n, 0.0) for n in names] for s in hits])
    p = model.predict_proba(X)[:, 1]
    return [s.t for s in hits], p, hits[0].label
