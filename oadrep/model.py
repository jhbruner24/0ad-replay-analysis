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


def _value_at(times, values, t):
    """Value at or just before time t. Times are ascending; values counter-like."""
    last = None
    for ti, vi in zip(times, values):
        if ti > t:
            break
        if isinstance(vi, (int, float)):
            last = vi
    return last


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


def build_samples(games, me, opp_aliases=None, ts=(180, 360, 540, 720, 900, 1200, 1500, 1800)):
    """One sample per (game, time-of-observation)."""
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
        for t in ts:
            if not mine.times or t > mine.times[-1]:
                continue
            feats = {"t_log": math.log(t + 1), "rating_diff": rating_diff, "rated": rated}
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


def train(samples):
    """Fit a calibrated logistic model on a set of samples."""
    X, y, _, names = _to_matrix(samples)
    base = Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=0.5, max_iter=1000))])
    # Calibrated wrapper: fits base on train folds, calibrates probabilities on
    # the held-out folds. Prefit=False so it does the split internally.
    model = CalibratedClassifierCV(base, method="isotonic", cv=5)
    model.fit(X, y)
    return model, names


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
