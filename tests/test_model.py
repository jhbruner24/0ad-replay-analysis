"""Tests for oadrep.model.

Uses the synthetic generator for the end-to-end signal-recovery test, so no
real 0 A.D. data is required. Requires numpy and scikit-learn.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy as np
    import sklearn  # noqa: F401
    SCI = True
except ImportError:
    SCI = False

from oadrep import compare, schema, synth
if SCI:
    from oadrep import model


class SelectFeatures(unittest.TestCase):
    """Feature selection must include decision stats and exclude combat ones."""

    def test_scalar_stat_prefix_matches_exact_key(self):
        seqs = {"populationCount": [1, 2], "unitsLost.Infantry": [0, 1]}
        if not SCI:
            self.skipTest("no scikit-learn")
        picked = model._select_features(seqs)
        self.assertIn("populationCount", picked)
        # unitsLost is outcome-coupled and must be excluded
        self.assertNotIn("unitsLost.Infantry", picked)

    def test_flattened_resource_stats_match_prefix(self):
        if not SCI:
            self.skipTest("no scikit-learn")
        seqs = {"resourcesGathered.food": [1], "resourcesGathered.wood": [2]}
        picked = model._select_features(seqs)
        self.assertEqual(set(picked), {"resourcesGathered.food",
                                       "resourcesGathered.wood"})

    def test_outcome_coupled_prefixes_are_excluded(self):
        # Second-level guard against future stats being reintroduced by mistake.
        if not SCI:
            self.skipTest("no scikit-learn")
        for bad in ("enemyUnitsKilled.Worker", "buildingsLost.Structure",
                    "lootCollected", "buildingsCapturedValue"):
            self.assertTrue(compare.is_outcome_coupled(bad),
                            f"{bad} should be outcome-coupled")
            picked = model._select_features({bad: [1, 2, 3]})
            self.assertNotIn(bad, picked)


class ValueAt(unittest.TestCase):
    def test_last_value_at_or_before_t(self):
        if not SCI:
            self.skipTest("no scikit-learn")
        self.assertEqual(model._value_at([0, 30, 60], [1, 2, 3], 45), 2)
        self.assertEqual(model._value_at([0, 30, 60], [1, 2, 3], 60), 3)
        self.assertIsNone(model._value_at([30, 60], [1, 2], 15))
        self.assertEqual(model._value_at([0, 30], [1, "junk"], 45), 1)


@unittest.skipUnless(SCI, "requires numpy + scikit-learn")
class EndToEndOnSyntheticCollection(unittest.TestCase):
    """Fit a real model on synthetic games and check it recovers the signal."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="oadrep-model-")
        synth.generate(cls.root, games=300, me="Me", opponents=("Friend",),
                       base_win_rate=0.5, drift=0.0, rated=1.0,
                       missing_metadata=0.0, unresolved=0.0, team_games=0.0,
                       versions=("0.27.0",), seed=13)
        cls.games = list(schema.iter_games(cls.root))
        # load_commands=False: synth doesn't write real cmd events, so command
        # features would all be zero and add noise. Stats-only is the pipeline
        # we can meaningfully test here.
        cls.samples = model.build_samples(cls.games, ["Me"], load_commands=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_every_decided_game_produces_samples(self):
        used = {s.game_dir for s in self.samples}
        decided = {g.directory for g in self.games if g.decided}
        # Every decided game reached at least one prediction time.
        self.assertEqual(used, decided)

    def test_features_include_time_and_rating(self):
        s = self.samples[0]
        self.assertIn("t_log", s.features)
        self.assertIn("rating_diff", s.features)
        # And the position features derived from the planted signal stats.
        self.assertIn("diff.income.food", s.features)
        self.assertIn("diff.alive.units.Infantry", s.features)
        self.assertIn("diff.material.army", s.features)

    def test_temporal_split_does_not_leak_games(self):
        train, test, _ = model.temporal_split(self.samples, holdout_frac=0.3)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(test), 0)
        self.assertTrue({s.game_dir for s in train}
                        .isdisjoint({s.game_dir for s in test}))

    def test_model_beats_baseline_on_planted_signal(self):
        train, test, _ = model.temporal_split(self.samples, holdout_frac=0.3)
        fit, names = model.train(train)
        result = model.evaluate(fit, test, names)
        base = result["base_rate"] * (1 - result["base_rate"])
        # The synthetic winner-eats-more-food signal is real; the model has to
        # beat "always predict the base rate" by a meaningful margin.
        self.assertLess(result["brier"], base * 0.75,
                        f"Brier {result['brier']:.3f} vs baseline {base:.3f}")
        self.assertGreater(result["auc"], 0.7)

    def test_reliability_report_shape(self):
        train, test, _ = model.temporal_split(self.samples, holdout_frac=0.3)
        fit, names = model.train(train)
        result = model.evaluate(fit, test, names)
        lines = model.reliability_report(result["labels"],
                                         result["predictions"]).splitlines()
        self.assertTrue(lines[0].startswith("  bucket"))
        self.assertGreater(len(lines), 3)

    def test_curve_for_game_returns_ordered_time_series(self):
        train, _, _ = model.temporal_split(self.samples, holdout_frac=0.3)
        fit, names = model.train(train)
        game_dir = train[0].game_dir
        ts, ps, label = model.curve_for_game(fit, self.samples, names, game_dir)
        self.assertIsNotNone(ts)
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(len(ts), len(ps))
        self.assertTrue(all(0 <= p <= 1 for p in ps))
        self.assertIn(label, (0, 1))

    def test_position_only_is_symmetric(self):
        mirrored = model.position_only(self.samples)
        self.assertEqual(len(mirrored), 2 * len(self.samples))
        a, b = mirrored[0], mirrored[1]
        self.assertEqual(a.label, 1 - b.label)
        for k, v in a.features.items():
            if k.startswith("diff."):
                self.assertEqual(b.features[k], -v)
        self.assertEqual(a.features["rating_diff"], 0.0)
        # A fully mirrored fit has nothing to lean on at t=0: P is ~0.5.
        fit, names = model.train(mirrored)
        X = np.array([[math.log(1) if n == "t_log" else 0.0 for n in names]])
        self.assertAlmostEqual(float(fit.predict_proba(X)[0, 1]), 0.5, delta=0.1)

    def test_export_matches_sklearn(self):
        train, test, _ = model.temporal_split(self.samples, holdout_frac=0.3)
        fit, names = model.train(train)
        exp = model.export_json(fit, names)
        X = np.array([[s.features.get(n, 0.0) for n in names] for s in test[:200]])
        want = fit.predict_proba(X)[:, 1]
        for s, w in zip(test[:200], want):
            self.assertAlmostEqual(model.predict_from_export(exp, s.features), w, places=9)

    def test_clipping_bounds_extrapolation(self):
        train, test, _ = model.temporal_split(self.samples, holdout_frac=0.3)
        fit, names = model.train(train, clip_quantile=0.1)
        exp = model.export_json(fit, names)
        self.assertEqual(len(exp["lo"]), len(names))
        feats = dict(test[0].features)
        big = {k: (v * 1e6 if k.startswith("diff.") else v) for k, v in feats.items()}
        clipped = {k: (min(max(v, lo), hi)) for (k, v), lo, hi
                   in zip(((n, big.get(n, 0.0)) for n in names), exp["lo"], exp["hi"])}
        self.assertAlmostEqual(model.predict_from_export(exp, big),
                               model.predict_from_export(exp, clipped), places=12)

    def test_live_features_prefer_exact_counts(self):
        spec = model.state_feature_spec()
        game = next(g for g in self.games if g.decided)
        mine, them = game.perspective(["Me"])
        def live(player, **exact):
            seqs = {"time": list(player.times)}
            for k, v in player.sequences.items():
                stat, _, sub = k.partition(".")
                if sub:
                    seqs.setdefault(stat, {})[sub] = list(v)
                else:
                    seqs[stat] = list(v)
            return {"sequences": seqs, **exact}
        a = model.live_features(spec, live(mine), live(them))
        b = model.live_features(spec, live(mine, popCount=1, classCounts={"Worker": 0}),
                                live(them, popCount=40, classCounts={"Worker": 30}))
        self.assertEqual(b["diff.state.pop"], -39.0)
        self.assertEqual(b["diff.alive.units.Worker"], -30.0)
        self.assertEqual(b["diff.material.army"], -30 * model.UNIT_COST["Worker"])
        self.assertEqual(a["diff.state.map_control"], b["diff.state.map_control"])

    def test_js_matches_python_end_to_end(self):
        """Run the mod's JS feature extraction + scorer under node against a
        live-shaped payload and the same export."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node not installed")
        spec = model.state_feature_spec()
        train, test, _ = model.temporal_split(self.samples, holdout_frac=0.3)
        fit, names = model.train(train, clip_quantile=0.1)
        exp = model.export_json(fit, names, spec)
        cases = []
        for game in [g for g in self.games if g.decided][:20]:
            mine, them = game.perspective(["Me"])
            def live(player, extra):
                seqs = {"time": list(player.times)}
                for k, v in player.sequences.items():
                    stat, _, sub = k.partition(".")
                    if sub:
                        seqs.setdefault(stat, {})[sub] = list(v)
                    else:
                        seqs[stat] = list(v)
                return {"sequences": seqs, **extra}
            cases.append([live(mine, {"popCount": 33, "classCounts": {"Worker": 20, "Cavalry": 2}}),
                          live(them, {"popCount": 41, "classCounts": {"Worker": 25}})])
        js = os.path.join(os.path.dirname(__file__), "..", "mod", "coach-overlay",
                          "gui", "session", "coach_overlay.js")
        script = (open(js, encoding="utf-8").read()
                  + "\nconst [model, cases] = JSON.parse(require('fs').readFileSync(0, 'utf8'));"
                  + "\nconsole.log(JSON.stringify(cases.map(([a, b]) =>"
                  + " CoachOverlay_Predict(model, CoachOverlay_Features(model, a, b)))));")
        out = subprocess.run([node, "-e", script], input=json.dumps([exp, cases]),
                             capture_output=True, text=True, check=True).stdout
        got = json.loads(out)
        self.assertEqual(len(got), len(cases))
        for (a, b), g in zip(cases, got):
            want = model.predict_from_export(exp, model.live_features(spec, a, b))
            self.assertAlmostEqual(g, want, places=9)


class PlayerIdMapping(unittest.TestCase):
    """The player_id on Player must match cmd <id> lines in commands.txt."""

    def test_ids_are_one_indexed_and_sequential(self):
        root = tempfile.mkdtemp(prefix="oadrep-pid-")
        try:
            synth.generate(root, games=1, me="Me", opponents=("Friend",),
                           missing_metadata=0.0, team_games=0.0,
                           unresolved=0.0, seed=1)
            games = list(schema.iter_games(root))
            self.assertEqual([p.player_id for p in games[0].players], [1, 2])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
