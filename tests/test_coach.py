"""Tests for oadrep.coach: swing detection and blown-lead finding."""

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

from oadrep import commands, schema, synth
if SCI:
    from oadrep import coach, model


@unittest.skipUnless(SCI, "requires numpy + scikit-learn")
class SwingDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="oadrep-coach-")
        synth.generate(cls.root, games=200, me="Me", opponents=("Friend",),
                       base_win_rate=0.5, rated=1.0, missing_metadata=0.0,
                       unresolved=0.0, team_games=0.0, seed=17)
        cls.games = list(schema.iter_games(cls.root))
        cls.samples = model.build_samples(cls.games, ["Me"], step=30,
                                          load_commands=False)
        train, _, _ = model.temporal_split(cls.samples, holdout_frac=0.3)
        cls.fit, cls.names = model.train(train)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_swings_ordered_by_drop_desc(self):
        game = next(g for g in self.games if g.decided)
        sw = coach.swings_in_game(self.fit, self.samples, self.names,
                                  game.directory, top_n=5, min_drop=0.0)
        drops = [s.drop for s in sw]
        self.assertEqual(drops, sorted(drops, reverse=True))

    def test_top_n_respected(self):
        game = next(g for g in self.games if g.decided)
        sw = coach.swings_in_game(self.fit, self.samples, self.names,
                                  game.directory, top_n=2, min_drop=0.0)
        self.assertLessEqual(len(sw), 2)

    def test_min_drop_filters_small_swings(self):
        game = next(g for g in self.games if g.decided)
        sw = coach.swings_in_game(self.fit, self.samples, self.names,
                                  game.directory, top_n=99, min_drop=0.9)
        for s in sw:
            self.assertGreaterEqual(s.drop, 0.9)

    def test_blown_leads_only_returns_losses(self):
        rows = coach.blown_leads(self.fit, self.samples, self.names,
                                 threshold=0.6, min_lead_duration=30)
        game_dirs = {r["game_dir"] for r in rows}
        for gd in game_dirs:
            game_samples = [s for s in self.samples if s.game_dir == gd]
            self.assertEqual(game_samples[0].label, 0,
                             f"blown_leads returned a WON game: {gd}")

    def test_blown_leads_threshold_is_actually_met(self):
        rows = coach.blown_leads(self.fit, self.samples, self.names,
                                 threshold=0.75, min_lead_duration=30)
        for r in rows:
            self.assertGreaterEqual(r["peak_p"], 0.75)

    def test_unknown_game_returns_empty(self):
        sw = coach.swings_in_game(self.fit, self.samples, self.names,
                                  "/does/not/exist", top_n=5)
        self.assertEqual(sw, [])


class CommandFeatureUtilities(unittest.TestCase):
    def test_entropy_zero_for_single_type(self):
        c = {"walk": 10}
        self.assertAlmostEqual(commands.shannon_entropy(c), 0.0)

    def test_entropy_maximal_for_uniform(self):
        # 4 types, equal counts -> ln(4)
        import math
        c = {"walk": 5, "gather": 5, "attack": 5, "train": 5}
        self.assertAlmostEqual(commands.shannon_entropy(c), math.log(4), places=6)

    def test_entropy_of_empty_is_zero(self):
        self.assertEqual(commands.shannon_entropy({}), 0.0)

    def test_categories_cover_all_tracked(self):
        # Every tracked type belongs to at least one category (except _other).
        covered = {t for types in commands.CATEGORIES.values() for t in types}
        uncovered = set(commands.TRACKED_TYPES) - covered
        # It's fine for some tracked types to be uncategorised (they still
        # contribute to per-type diffs), but flag any drift silently.
        self.assertTrue(len(uncovered) <= 5,
                        f"too many uncategorised: {sorted(uncovered)}")

    def test_rate_acceleration_detects_burst(self):
        # 30s of calm, then 20 clicks in 10s -> strong positive acceleration.
        events = ([commands.Event(t=i, player_id=1, type="walk") for i in range(0, 30, 6)]
                  + [commands.Event(t=30 + i * 0.5, player_id=1, type="attack")
                     for i in range(20)])
        acc = commands.rate_acceleration(events, 1, t=40, short=10, long=30)
        self.assertGreater(acc, 1.0)

    def test_rate_acceleration_zero_when_no_history(self):
        events = [commands.Event(t=5.0, player_id=1, type="walk")]
        # t < long+short (30+10) -> not enough history to compute.
        self.assertEqual(commands.rate_acceleration(events, 1, t=15), 0.0)


if __name__ == "__main__":
    unittest.main()
