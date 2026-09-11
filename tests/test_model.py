"""Tests for oadrep.model.

Uses the synthetic generator for the end-to-end signal-recovery test, so no
real 0 A.D. data is required. Requires numpy and scikit-learn.
"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy  # noqa: F401
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
        # And at least one differential from the planted signal stat family.
        self.assertTrue(any(k.startswith("diff.resourcesGathered.")
                            for k in s.features))

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
