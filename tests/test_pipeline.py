"""End-to-end tests over synthetic data. Standard library only: python3 -m unittest discover tests"""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oadrep import compare, schema, synth


class SyntheticCollection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="oadrep-test-")
        cls.manifest = synth.generate(cls.root, games=120, drift=0.0, seed=42)
        cls.games = list(schema.iter_games(cls.root))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_every_replay_directory_is_found(self):
        self.assertEqual(len(self.games), len(self.manifest))

    def test_games_are_ordered_chronologically(self):
        orders = [g.order for g in self.games]
        self.assertEqual(orders, sorted(orders))

    def test_missing_metadata_is_reported_not_raised(self):
        broken = [g for g in self.games if any("metadata" in p for p in g.problems)]
        expected = sum(not m["has_metadata"] for m in self.manifest)
        self.assertEqual(len(broken), expected)
        for game in broken:
            self.assertEqual(game.players, [])       # degrades, does not crash

    def test_gaia_is_excluded_from_players(self):
        for game in self.games:
            self.assertNotIn("Gaia", [p.name for p in game.players])

    def test_nested_resource_stats_are_flattened(self):
        playable = next(g for g in self.games if g.players)
        seqs = playable.players[0].sequences
        self.assertIn("resourcesGathered.food", seqs)
        self.assertIn("populationCount", seqs)
        self.assertNotIn("time", seqs)               # time is held separately

    def test_series_are_parallel_to_times(self):
        for game in self.games:
            for player in game.players:
                for stat, series in player.sequences.items():
                    self.assertEqual(len(series), len(player.times),
                                     f"{stat} not parallel to time")

    def test_unresolved_games_are_not_counted_as_decided(self):
        for game in self.games:
            if game.players and all(p.state == "active" for p in game.players):
                self.assertFalse(game.decided)

    def test_perspective_returns_both_sides(self):
        game = next(g for g in self.games if g.is_1v1)
        me, them = game.perspective("Me")
        self.assertEqual(me.name, "Me")
        self.assertNotEqual(them.name, "Me")
        self.assertIsNone(game.perspective("Nobody"))


class SignalRecovery(unittest.TestCase):
    """The generator plants a known advantage; the analysis must find it."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="oadrep-signal-")
        synth.generate(cls.root, games=160, base_win_rate=0.5, drift=0.0,
                       missing_metadata=0.0, team_games=0.0, unresolved=0.0, seed=7)
        games = list(schema.iter_games(cls.root))
        data, cls.used = compare.collect(games, "Me")
        cls.findings = compare.analyse(data, seed=7)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_all_games_usable(self):
        self.assertEqual(self.used, 160)

    def test_planted_food_signal_is_recovered(self):
        hit = next((f for f in self.findings
                    if f["stat"] == "resourcesGathered.food"
                    and f["bucket"] == (300, 600)), None)
        self.assertIsNotNone(hit, "planted signal window not evaluated")
        self.assertTrue(hit["significant"], "planted signal not detected")
        self.assertGreater(hit["effect"], 0.5, "signal detected in wrong direction")
        self.assertGreater(hit["median_won"], hit["median_lost"])

    def test_planted_signal_ranks_above_pure_noise(self):
        """Wood has no planted advantage; food should outrank it."""
        def best(stat):
            vals = [abs(f["effect"] - 0.5) for f in self.findings if f["stat"] == stat]
            return max(vals) if vals else 0.0
        self.assertGreater(best("resourcesGathered.food"),
                           best("resourcesGathered.wood"))

    def test_effect_sizes_are_probabilities(self):
        for f in self.findings:
            self.assertGreaterEqual(f["effect"], 0.0)
            self.assertLessEqual(f["effect"], 1.0)
            lo, hi = f["ci"]
            self.assertLessEqual(lo, hi)


class EffectSizeMath(unittest.TestCase):
    def test_identical_samples_are_chance(self):
        self.assertEqual(compare.common_language_effect([1, 2, 3], [1, 2, 3]), 0.5)

    def test_total_separation_is_one(self):
        self.assertEqual(compare.common_language_effect([10, 11], [1, 2]), 1.0)

    def test_empty_input_is_none(self):
        self.assertIsNone(compare.common_language_effect([], [1]))

    def test_bucket_takes_last_value_in_window(self):
        times = [30.0, 60.0, 330.0, 360.0]
        vals = [1, 2, 3, 4]
        self.assertEqual(compare.value_in_bucket(times, vals, 0, 300), 2)
        self.assertEqual(compare.value_in_bucket(times, vals, 300, 600), 4)
        self.assertIsNone(compare.value_in_bucket(times, vals, 600, 900))


if __name__ == "__main__":
    unittest.main()
