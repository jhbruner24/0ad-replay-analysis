"""Turn a fitted win-probability model into coaching output.

The idea from the design doc: swing detection. Find the moments in a game where
P(win) dropped hardest, and surface the state at those moments. That's a
value-function delta - what chess engines actually give - not a claim about
alternate timelines.

Two report tiers:

- `swings_in_game`: within one game, the N largest single-step drops in P(win).
  Where did the tide turn.
- `blown_leads`: across a collection, the games where you had P >= threshold and
  still lost. These are the highest-value teaching moments because the loss
  came from something specific, not from being behind.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import model as model_mod


@dataclass
class Swing:
    t_before: float          # seconds into game, sample before the drop
    t_after: float           # seconds into game, sample after
    p_before: float
    p_after: float
    drop: float              # p_before - p_after; positive = worse for me
    top_deltas: list         # [(feature_name, my_change), ...] largest movers


def _coef_vector(fit, names):
    """Averaged, standardised logistic coefficients from the calibrated model.

    Attributing a probability change to raw feature movement (my earlier
    approach) is misleading: cumulative counters like `resourcesGathered.food`
    always dominate in absolute delta terms but the model may weight them
    little. Multiplying delta by the standardised coefficient gives what the
    model itself was reacting to.
    """
    inner = getattr(fit, "inner", fit)
    coefs = []
    for cc in inner.calibrated_classifiers_:
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None and hasattr(est, "coef_"):
            coefs.append(est.coef_[0])
    if not coefs:
        return None
    avg = np.mean(coefs, axis=0)
    sd = getattr(fit, "sd", np.ones_like(avg))
    return avg / sd                                   # unstandardise: coef per raw feature unit


def _driving_features(samples_before, samples_after, coef, names, k=5):
    """Features ranked by their contribution to the logit change.

    Contribution = (feature delta) * coefficient. Positive contribution moves
    P(win) toward 1; negative moves it toward 0. We show the negative
    contributions (things that dragged the probability DOWN) since those are
    the moves that hurt.
    """
    if coef is None:
        return []
    idx = {n: i for i, n in enumerate(names)}
    contribs = []
    for name in set(samples_before.features) | set(samples_after.features):
        if name not in idx:
            continue
        delta = (samples_after.features.get(name, 0.0)
                 - samples_before.features.get(name, 0.0))
        if delta == 0:
            continue
        contribs.append((name, delta * coef[idx[name]], delta))
    # Most-hurtful first (largest negative logit contribution)
    contribs.sort(key=lambda x: x[1])
    return contribs[:k]


def swings_in_game(fit, samples, names, game_dir, top_n=5, min_drop=0.05):
    """Return the largest single-step probability drops in one game.

    A "drop" is worst for the perspective player - the win probability fell.
    Rank by drop magnitude and return the top_n exceeding min_drop.
    """
    game_samples = [s for s in samples if s.game_dir == game_dir]
    game_samples.sort(key=lambda s: s.t)
    if len(game_samples) < 2:
        return []
    X = np.array([[s.features.get(n, 0.0) for n in names] for s in game_samples])
    p = fit.predict_proba(X)[:, 1]
    coef = _coef_vector(fit, names)
    swings = []
    for i in range(1, len(p)):
        drop = float(p[i - 1] - p[i])
        if drop < min_drop:
            continue
        swings.append(Swing(
            t_before=game_samples[i - 1].t, t_after=game_samples[i].t,
            p_before=float(p[i - 1]), p_after=float(p[i]), drop=drop,
            top_deltas=_driving_features(game_samples[i - 1], game_samples[i],
                                         coef, names),
        ))
    swings.sort(key=lambda s: -s.drop)
    return swings[:top_n]


def blown_leads(fit, samples, names, threshold=0.85, min_lead_duration=60):
    """Games where you held P >= threshold for at least `min_lead_duration` seconds
    and still lost. Ranked by peak probability.
    """
    per_game = {}
    for s in samples:
        per_game.setdefault(s.game_dir, []).append(s)
    losses = []
    for gd, gs in per_game.items():
        gs.sort(key=lambda s: s.t)
        if not gs or gs[0].label != 0:                        # not a loss
            continue
        X = np.array([[s.features.get(n, 0.0) for n in names] for s in gs])
        p = fit.predict_proba(X)[:, 1]
        # Longest run of consecutive samples with p >= threshold.
        best_run, run_start_t, run_start_i = 0.0, None, None
        peak, peak_t = 0.0, None
        run_len = 0
        for i, (s, prob) in enumerate(zip(gs, p)):
            if prob >= threshold:
                run_len += 1
                if run_len == 1:
                    run_start_t = s.t
                    run_start_i = i
                dur = s.t - run_start_t
                if dur > best_run:
                    best_run = dur
                if prob > peak:
                    peak, peak_t = float(prob), s.t
            else:
                run_len = 0
        if best_run >= min_lead_duration:
            losses.append({
                "game_dir": gd, "order": gs[0].order,
                "peak_p": peak, "peak_t": peak_t,
                "lead_duration_s": best_run,
                "game_length_min": gs[-1].t / 60.0,
            })
    losses.sort(key=lambda d: -d["peak_p"])
    return losses


def format_swing(swing: Swing) -> str:
    lines = [f"  t={swing.t_before/60:.1f}min -> {swing.t_after/60:.1f}min   "
             f"P(win) {swing.p_before:.0%} -> {swing.p_after:.0%}   "
             f"(drop {swing.drop:.0%})"]
    for row in swing.top_deltas:
        if len(row) == 2:                              # legacy raw-delta format
            name, delta = row
            sign = "+" if delta >= 0 else ""
            lines.append(f"      {name:<34} {sign}{delta:>10,.1f}")
        else:
            name, contrib, delta = row
            lines.append(f"      {name:<34} logit {contrib:+.2f}   "
                         f"(feature moved {delta:+,.1f})")
    return "\n".join(lines)


def format_blown_leads(rows) -> str:
    if not rows:
        return "  (none)"
    lines = [f"  {'peak':<6} {'at':<8} {'lead held':<12} {'game len':<10} game"]
    lines.append("  " + "-" * 76)
    for r in rows:
        peak_t = f"{r['peak_t']/60:.1f}min"
        held = f"{r['lead_duration_s']/60:.1f}min"
        length = f"{r['game_length_min']:.1f}min"
        gid = r['game_dir'].split('/')[-1]
        lines.append(f"  {r['peak_p']:>5.0%}  {peak_t:<8} {held:<12} {length:<10} {gid}")
    return "\n".join(lines)
