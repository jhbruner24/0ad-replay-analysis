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
- Findings above were read from the engine source as of the 2024 GitHub mirror
  (roughly a27) and re-verified where possible. If a field name looks wrong on a
  newer release, `census.py --debug` prints the raw keys.

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

Early. `census.py` works and is verified against a synthetic fixture; it has not
yet been run against a large real collection. Win-probability modelling and the
per-match visualisation are not built yet.

Issues and PRs welcome, particularly from anyone with a replay collection to test
against.

## License

MIT — see [LICENSE](./LICENSE). Note that this project only *reads* files the game
produces; it contains no 0 A.D. code. Anything that later ships as a game mod
would need to consider the engine's own licensing.
