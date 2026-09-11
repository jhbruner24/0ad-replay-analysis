# 0 A.D. Replay Analysis

Tools for statistical analysis of [0 A.D.](https://play0ad.com) replays — match
inventory today, win-probability modelling next.

Everything here reads files the game has already written. No engine build, no mod,
no re-simulation, no dependencies outside the Python standard library.

## The useful thing to know about 0 A.D. replays

A replay's `commands.txt` is a **command log, not a state log**. It records what
players clicked — `start`, `turn`, `cmd`, `hash`, `end` — and never what the board
looked like. Nothing in it says "player 1 had 47 workers at 6:00". Reconstructing
state from it means re-running the deterministic simulation.

**But you usually don't have to.** The `metadata.json` written alongside it already
contains a per-player time series:

- `StatisticsTracker` samples every statistic it keeps every **30 seconds** of game
  time onto a `sequences` array
- `GuiInterface.GetExtendedSimulationState()` attaches those sequences to each
  player
- `GetReplayMetadata()` returns them, and `CReplayLogger::SaveMetadata` writes the
  result to disk at the end of every completed game

So a surprising amount of analysis needs no engine interaction at all.

### What's in the sequences

Sampled every 30 seconds, per player, column-oriented (parallel arrays):

| Group | Fields |
|---|---|
| Units | `unitsTrained`, `unitsLost`, `enemyUnitsKilled`, `unitsCaptured` (+ `…Value`) |
| Buildings | `buildingsConstructed`, `buildingsLost`, `enemyBuildingsDestroyed`, `buildingsCaptured` (+ values) |
| Economy | `resourcesCount`, `resourcesGathered`, `resourcesUsed`, `resourcesSold`, `resourcesBought`, `tradeIncome`, `tributesSent`/`Received`, `treasuresCollected`, `lootCollected` |
| Map | `percentMapExplored`, `percentMapControlled`, `peakPercentMapControlled` (+ team variants) |
| Other | `populationCount`, `successfulBribes`, `failedBribes` |

Resource-typed stats nest one level: `resourcesGathered: {food: [...], wood: [...]}`.

### What's only an end-of-game snapshot

Present in `playerStates` but *not* sequenced over time: `phase`,
`researchedTechs`, `classCounts` and `typeCountsByClass` (army composition),
`resourceGatherers` (workers by resource), `popCount`, `civ`, and `state` — the
last being the win/loss label.

`playerStates` covers every player, so **one player's replay carries full
two-sided data**.

### Caveats

- **Replays are version-locked.** A replay only reproduces on the engine version
  and mod set that recorded it — that's what the `hash` lines verify. A collection
  spanning releases also spans balance changes, which is a confound for any model
  trained across it.
- **Unfinished games write no `metadata.json` at all.** It's saved from
  `CGame::~CGame()`, gated on the game having started, so a crash or a hard quit
  leaves the directory without it.
- **Lobby games append the rating to the nickname**, as `Alec576 (1323)`. The same
  person therefore appears under a different name at every rating they have held.
  Split with the engine's own pattern (`gui/common/gamedescription.js`), which
  this package copies verbatim. The rating is worth keeping: it is a direct skill
  measure and a strong candidate feature.
- The map name is `settings.mapName` on the `start` line, not in `metadata.json`.
- Some replays carry **no `engine_version`** in the start line. The parent
  directory is named for the version that wrote it, so it serves as a fallback.
- Collections contain **games left running** rather than played — hours long, with
  hundreds of snapshots. They distort time buckets and bloat exports; filter them.
- Findings above were read from the engine source as of the 2024 GitHub mirror
  (roughly a27). If a field name looks wrong on a newer release,
  `python3 -m oadrep.schema <replay-dir>` checks each assumption individually.

## Layout

| Path | What it does |
|---|---|
| `oadrep/schema.py` | **The only module that knows the file formats.** Every assumption is numbered and documented; a release that renames a field changes this file and nothing else. |
| `oadrep/synth.py` | Generates a synthetic replay collection — messy by default, with a *known* planted signal so analysis code can be checked against ground truth. |
| `oadrep/compare.py` | Wins-vs-losses trajectory comparison. |
| `census.py` | Collection inventory (CLI). |
| `tests/` | End-to-end over synthetic data. `python3 -m unittest discover -s tests` |

Developing without a replay collection to hand:

```bash
python3 -m oadrep.synth /tmp/fake --games 200 --drift 0.35
python3 census.py --root /tmp/fake --me Me
python3 -m oadrep.compare --root /tmp/fake --me Me
```

Checking the schema assumptions against one real replay:

```bash
python3 -m oadrep.schema ~/.local/share/0ad/replays/0.27.0/<some-game>/
```

## census.py

Inventories a replay collection. Run this before modelling anything — the number
of replay directories on disk is not the number of usable games.

```bash
python3 census.py                          # auto-detects the replay directory
python3 census.py --me "YourName"          # adds win/loss and drift analysis
python3 census.py --me "YourName" --csv games.csv
python3 census.py --debug                  # dump raw JSON keys of the first game
```

Run it once without `--me`; it prints the player names it found so you can pass
the right one.

It reports:

- Replay directories found, and how many carry usable metadata
- Engine-version, civ-matchup and map spread
- Players per game, game-length distribution
- Sequence resolution (confirms the 30-second cadence on your version)
- Head-to-head win rate, and the baseline a model would have to beat
- **A chronological drift check** — win rate across time blocks

`--csv` writes long format (`game, order, player, civ, state, t_seconds, stat,
value`), ready to load into pandas or R.

It is strictly read-only and never writes into the replay directory.

## compare.py — wins vs losses

```bash
python3 -m oadrep.compare --me "YourName"
```

Splits each of your statistics into games you won and games you lost, per time
window, and reports where the two differ. Not a model: at a few hundred games it
is better powered than anything learned, needs no training split, and its output
is directly actionable.

Comparison is by **common-language effect size** — the probability that a randomly
chosen won game exceeds a randomly chosen lost game — computed from the
Mann-Whitney U statistic, with a bootstrap confidence interval. Rank-based, so it
needs no normality assumption and a couple of blowouts do not move it.

Results are split into two tables, which matters more than it sounds:

- **Decision statistics** — economy, expansion, production. Things you chose.
- **Outcome-coupled statistics** — combat results. `enemyUnitsKilled` separates
  wins from losses almost perfectly, but "you killed more in the games you won" is
  a restatement of winning, not advice. Reported separately so tautologies do not
  crowd out the coachable findings.

Dozens of stat/window pairs are tested, so some intervals exclude chance by luck.
The output is a ranked list of leads, not confirmed findings.

## Notes on modelling this data

Recorded here because they're easy to get wrong and cost real time:

- **Effective sample size is the number of games, not the number of rows.** A
  20-minute game yields ~40 snapshots, so 200 games looks like 8,000 rows — but
  there are only 200 independent outcomes. Flexible models will memorise which
  game a row came from and validate beautifully on nothing.
- **Split by game, never by timestep.** A random row split leaks the outcome.
- **If the collection is one recurring matchup, split chronologically.** Two
  players who adapt to each other are a non-stationary process; grouped k-fold
  leaks the future into the past.
- **The baseline is the observed win rate, not 50%.**
- **Optimise calibration, not accuracy** — log loss, Brier score, reliability
  diagrams. A probability display that says 70% should be right 70% of the time.
- **Late-game features are self-fulfilling.** "Opponent has lost 80% of their
  buildings" predicts the winner at 99% and teaches nothing. The useful signal is
  early.

## Status

Early, and honest about it: **everything here has been developed against synthetic
data.** The schema was read from the 2024 GitHub source mirror, so field names are
verified against roughly a27 rather than the current release. Nothing has yet met
a real replay collection.

That is why all format knowledge sits in `oadrep/schema.py` behind numbered
assumptions — if a name has changed, one module changes and the analysis on top of
it does not.

**Update:** first contact with a real ~1,800-replay collection corrected three
schema assumptions (map field, version fallback, nickname ratings). All three were
one-module fixes, which was the point of the structure. The synthetic generator
now reproduces those quirks so the tests cover them.

Built: collection census, schema adapter, synthetic generator, wins-vs-losses
comparison, 22 tests. Not built: win-probability model, per-match visualisation.

Issues and PRs welcome, particularly from anyone with a replay collection to test
against.

## License

MIT — see [LICENSE](./LICENSE). Note that this project only *reads* files the game
produces; it contains no 0 A.D. code. Anything that later ships as a game mod
would need to consider the engine's own licensing.

## Live overlay mod

`mod/coach-overlay/` is the 0 A.D. GUI mod that publishes
`GetExtendedSimulationState` every 5 s. Install by copying (or symlinking)
the directory into your 0 A.D. `mods/` folder, then enable it in
Settings → Mod Selection. It writes to
`saves/campaigns/coach-overlay/live_state.json` under the user data
directory — GUI-context `Engine.WriteJSONFile` is restricted to a few
prefixes and `saves/campaigns/` is one of them. `live_coach.py` reads
that path if you want the number in a terminal too.

The mod scores the game itself: `export_model.py` fits the model on the
replay collection and writes the coefficients and isotonic calibration to
`mod/coach-overlay/gui/coach-overlay/model.json`; `coach_overlay.js`
re-implements the scoring (`oadrep.model.predict_from_export` is the
reference; a test runs the JS under node and checks parity). The panel
draws "Win NN%" with a bar in the top panel, right of the civ icon, for
the viewed player, 1v1 only. Re-export when the collection grows:

    python3 export_model.py --me wace8000 --opp wolfmapking,noob5layer --install

Full quit + relaunch of 0 A.D. is needed after any change to the mod: the
VFS does not notice changed files on macOS.
