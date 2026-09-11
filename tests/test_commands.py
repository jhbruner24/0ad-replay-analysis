"""Tests for oadrep.commands. Standard-library only."""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oadrep import commands


def _write(commands_txt):
    fh = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")
    fh.write(commands_txt)
    fh.close()
    return fh.name


class ParseCommands(unittest.TestCase):
    def test_timing_accumulates_from_turn_lengths(self):
        # Two 500ms turns then a cmd; the cmd should land at 1.0s.
        path = _write(
            'start {"engine_version":"0.27.0"}\n'
            'turn 0 500\nend\n'
            'turn 1 500\nend\n'
            'turn 2 500\ncmd 1 {"type":"gather"}\nend\n'
        )
        events = commands.parse(path)
        os.unlink(path)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].t, 1.5, places=3)
        self.assertEqual(events[0].player_id, 1)
        self.assertEqual(events[0].type, "gather")

    def test_variable_turn_lengths_are_respected(self):
        # If the game slows down, subsequent commands sit at the correct time.
        path = _write(
            'start {}\n'
            'turn 0 500\nend\n'
            'turn 1 2000\ncmd 2 {"type":"train"}\nend\n'
        )
        events = commands.parse(path)
        os.unlink(path)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].t, 2.5, places=3)

    def test_malformed_lines_are_skipped_not_raised(self):
        path = _write(
            'start {}\n'
            'turn 0 500\n'
            'cmd not-an-integer {"type":"walk"}\n'
            'cmd 1 {not valid json}\n'
            'cmd 1 {"type":"walk"}\n'
            'end\n'
        )
        events = commands.parse(path)
        os.unlink(path)
        self.assertEqual([e.type for e in events], ["walk"])

    def test_counts_in_window_filters_by_player_and_time(self):
        events = [
            commands.Event(t=1.0, player_id=1, type="walk"),
            commands.Event(t=5.0, player_id=1, type="attack"),
            commands.Event(t=5.0, player_id=2, type="attack"),
            commands.Event(t=15.0, player_id=1, type="train"),
        ]
        c = commands.counts_in_window(events, player_id=1, t_start=0, t_end=10)
        self.assertEqual(c["walk"], 1)
        self.assertEqual(c["attack"], 1)
        self.assertEqual(c["train"], 0)                    # outside window
        self.assertNotIn("attack", commands.counts_in_window(events, 2, 6, 10))

    def test_untracked_types_are_bucketed_as_other(self):
        events = [commands.Event(t=1.0, player_id=1,
                                 type="wildly-experimental-command")]
        c = commands.counts_in_window(events, 1, 0, 5)
        self.assertEqual(c["_other"], 1)

    def test_missing_file_returns_empty_list(self):
        self.assertEqual(commands.parse("/nonexistent/commands.txt"), [])


if __name__ == "__main__":
    unittest.main()
