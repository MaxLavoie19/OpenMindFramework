# entrypoint

## Purpose

Every way to run the framework. Entrypoints handle input, output and where logs go; the domains do the work.

## Content

| File | What it is |
|---|---|
| `play.py` | `openmind-play`: play a game in the terminal, as humans or with the agent |
| `solve.py` | `openmind-solve`: solve a game's legal actions from where it starts with the CSP alone and print the solutions |
| `rollout_options.py` | `add_rollout_limit_options(parser)` and `checked_unfinished_payoff(parser, arguments)`: the rollout limit and its unfinished payoff, which must be given together |
| `dashboard.py` | `openmind-dashboard`: serve a page following a value training while it runs, on the machine it runs on |
| `rerun_call.py` | `openmind-rerun-call`: run again, alone, the call a worker ended on for its memory cap, and show where it holds memory |

## `openmind-rerun-call`

```bash
.venv/bin/openmind-rerun-call <log directory>/chess/memory/<YYYY-MM-DD_HH-MM-SS>-worker-<pid>.txt
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 .venv/bin/openmind-rerun-call <diagnosis>   # capped
```

| Option | Default | Meaning |
|---|---|---|
| `--lines N` | `20` | source lines shown, those holding the most memory first |
| `--log-directory DIR` | `data/log/rerun-call` | where logs are saved |
| `--debug-session FILE` | none | a debug session file (see `debug/README.md`): verbosity, breakpoints, what time does while paused; with it, Ctrl+C pauses at the next reasoning frame |

Given a memory diagnosis or the `.pickle` beside it (see `parallel/README.md`), it loads the call, from the project that
ran it so its domain can be imported, runs it in this process under `tracemalloc`, and prints and logs:

```
The call ended|raised after <seconds> seconds; traced memory peaked at <bytes> bytes and holds <bytes> bytes with the call's objects still alive
bytes  blocks  line
<bytes>  <blocks>  <file>:<line>
```

The lines are those holding memory once the call is over, with the call's function, its caches included, still alive;
the peak says how much the call held at most. Tracing makes the call slower and bigger than it was in the worker, so a
call that went over a cap is best run again under a capped scope. A pickled call no path points to is rejected.

## `openmind-dashboard`

```bash
.venv/bin/openmind-dashboard chess     # from the project that trains, on the training machine
```

| Option | Default | Meaning |
|---|---|---|
| `--host ADDRESS` | the machine's Tailscale IPv4 address, from `tailscale ip -4` | address to listen on; without Tailscale, give it, knowing whoever reaches it sees the page |
| `--port N` | `8765` | port to listen on |
| `--refresh N` | `30` | seconds between the page's reloads |
| `--log-directory DIR` | `data/log/train-values` | where training logs are, under `<domain>/` |
| `--syslog PATH` | `/var/log/syslog` | the system log earlyoom writes to; `none` skips it |
| `--dashboard-log-directory DIR` | `data/log/dashboard` | where the dashboard's own log is saved |
| `--debug-session FILE` | none | a debug session file (see `debug/README.md`): verbosity, breakpoints, what time does while paused; with it, Ctrl+C pauses at the next reasoning frame |

Every request takes a fresh snapshot (see `dashboard/README.md`) and serves it as one page: the current round with its
self-play games and deduced moves, whether the training runs and with how many workers, memory and swap, each training
process's memory, the latest notable log lines, and earlyoom's latest kills. It runs until stopped; no address can be
found without Tailscale and `--host`, which is rejected.

## `openmind-play`

```bash
.venv/bin/openmind-play tictactoe                                 # two humans
.venv/bin/openmind-play tictactoe --agent O                       # human X against the agent
.venv/bin/openmind-play tictactoe --agent X --agent O --seed 7    # the agent against itself
.venv/bin/openmind-play tictactoe/fourinarow --agent O            # 4 in a row, a variant of tic-tac-toe
```

| Option | Default | Meaning |
|---|---|---|
| `--agent PLAYER` | none | a player the agent controls; repeat for several |
| `--iterations N` | `1000` | MCTS iterations per agent move |
| `--seed S` | unseeded | random seed for the agent's search |
| `--rollout-limit N` | no limit | actions an agent's rollout plays at most before every player gets the unfinished payoff |
| `--unfinished-payoff X` | none | each player's payoff for a rollout stopped at the limit; needed with `--rollout-limit`, and rejected without it |
| `--time-control MINUTES+SECONDS` | no clock | every player, human or agent, plays on a clock, a human's thinking at the prompt included; both clocks are printed before each move, and a player whose time runs out loses as the domain's timeout rule says, the game ending with `<player>'s time ran out`; written as chess writes a time control: `3+2` is 3 minutes and 2 seconds a move; `--iterations` then no longer counts, each move's budget replacing it; a domain without a timeout rule is refused, and so is anything but minutes+seconds or a base of 0 |
| `--expected-steps N` | `30` | with a clock: steps a player expects to be left at any point of a game, the plain time budget estimator sharing the time left between them (see `timing/README.md`); below 1 is refused |
| `--time-reserve X` | `0.05` | with a clock: the share of a player's base time kept in reserve; below it the player plays only random moves, so its clock never runs out; from 0 to below 1 |
| `--selection ucb1\|puct` | `ucb1` | how a tried node picks the action to follow: `ucb1` tries every legal action once first, `puct` follows Q + c · P · √N / (1 + n) with a prior (see `mcts/README.md`) |
| `--prior uniform\|rater\|value` | `uniform` | the prior PUCT follows: every action alike, the agent's rules' ratings, or its value rules' values of each action's outcomes; only `uniform` here, the agent having no rules when playing |
| `--puct-exploration X` | `1.5` | PUCT's exploration weight c; 0 or more |
| `--prior-temperature X` | `0.1` | the softmax temperature turning ratings or values into a prior, lower following the best more closely; above 0 |
| `--log-level LEVEL` | `INFO` | lowest level saved in the game log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/play` | where game logs are saved |
| `--debug-session FILE` | none | a debug session file (see `debug/README.md`): verbosity, breakpoints, what time does while paused; with it, Ctrl+C pauses at the next reasoning frame |

1. Prints the state as the player to act sees it: each two-dimensional grid under its name and column numbers, with
   `.` for an empty cell, then one `name = value` line per other model (`GridTextMapper`, see `world/README.md`).
2. On a human's turn, lists the legal actions by number and reads the number of the action to perform. Anything else
   asks again; end of input (Ctrl+D) ends the session.
3. On an agent's turn, the agent searches and the CLI prints `<player> chose <action>`.
4. The game's RBS gives the outcome distribution; when there are several outcomes, one is drawn by its probability.
5. When no action is legal, prints the final state, payoffs included.

Games are declared by name into the knowledge base through `agent/factory/game_factory.py`: `tictactoe`, its variants
`tictactoe/fourinarow` and `tictactoe/gomoku`, `sudoku`, `prisonersdilemma` and its variants, `rockpaperscissors`,
and any game an installed project registers (see `agent/README.md`). An `--agent` player the game doesn't have is
rejected.

Each session writes `<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, from `--log-level` up (a variant's logs go
in a folder per variant, such as `data/log/play/tictactoe/fourinarow/`), as
`LEVEL logger message` lines. At `INFO`: the agent's search results (see `mcts/README.md`), plus, from logger
`openmind.entrypoint.play`:

- `INFO Playing <domain>`
- `INFO Chose <action>`
- `INFO No legal action left: game over`
- `INFO Input ended before the game was over`

At `DEBUG`, every solver candidate, predictor effect and search iteration is added, including those inside the
agent's rollouts.

## `openmind-solve`

```bash
.venv/bin/openmind-solve sudoku
.venv/bin/openmind-solve sudoku --limit 1 --log-level INFO
.venv/bin/openmind-solve sudoku/euler sudoku/top95 sudoku/hardest
.venv/bin/openmind-solve sudoku/top95/7
```

| Argument or option | Default | Meaning |
|---|---|---|
| `DOMAIN ...` | required | one or more of: a domain (`sudoku`), a sudoku collection (`sudoku/<collection>`), or one of its puzzles (`sudoku/<collection>/<number>`, counting from 1) |
| `--limit N` | `2` | most solutions to find per domain or puzzle; 2 is enough to tell whether a solution is unique |
| `--log-level LEVEL` | `DEBUG` | lowest level saved in the log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/solve` | where logs are saved |
| `--debug-session FILE` | none | a debug session file (see `debug/README.md`): verbosity, breakpoints, what time does while paused; with it, Ctrl+C pauses at the next reasoning frame |
| `--puzzle-directory DIR` | `data/sudoku` | where sudoku collections are read from, one `<collection>.txt` file each (the root README has the downloads) |

Every DOMAIN is resolved before anything is solved. An unknown domain raises `ValueError`; a collection missing from
the puzzle directory is rejected with the collections found; a puzzle number outside its collection is rejected with
the collection's range.

One solver solves every DOMAIN in turn from its initial state (see `csp/README.md`):

- **A domain or a single puzzle:** for each solution, the predictor applies the action and the CLI prints
  `Solution <n> (probability <p>):` followed by the resulting state, the grid laid out under its column numbers, then
  one `name = value` line per other model (`GridTextMapper`). A summary line follows.
- **A collection:** a summary line per puzzle, in file order, without the solutions, then the totals:
  `<collection name>: <n> puzzle(s), <n> solution(s), <a> assignments, <d> dead ends, <p> values pruned, <seconds> seconds`.

A summary line is `<name>: <n> solution(s), <a> assignments, <d> dead ends, <p> values pruned, <seconds> seconds`,
adding ` (limit reached)` when the search stopped at the limit. Names are `sudoku`, `sudoku/<collection>` and
`sudoku/<collection>/<number>`. The counts come from `Solver.solve_with_statistics`; a domain given twice in one run
comes from the cache the second time and repeats the first search's counts.

Each DOMAIN writes `<log directory>/<name>/<YYYY-MM-DD_HH-MM-SS>.log`. From logger `openmind.entrypoint.solve`, a
collection's log starts with `INFO Solving <collection name>` and ends with its totals line. Each domain or puzzle
logs `INFO Solving <name>`, then, at the default `DEBUG`, the solver's trace (tries, dead ends, summary) and, when its
solutions are printed, the predictor's effects, then its summary line at `INFO`.

## Notes

- End-to-end tests: `test/end_to_end/play_tictactoe_tests.py`, `test/end_to_end/play_fourinarow_tests.py`,
  `test/end_to_end/solve_sudoku_tests.py`.
