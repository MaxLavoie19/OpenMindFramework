# OpenMind Framework

An AI framework in which an agent works within a domain by searching with Monte-Carlo Tree Search (MCTS). A constraint
satisfaction problem generates the possible actions and their parameters, a predictor gives the probability of each
outcome, and a strategy rule-based system distills best play, weighted by expected value. The domains so far are
tic-tac-toe with its variants (4 in a row, gomoku) and sudoku. The framework grows in small increments;
[doc/architecture.md](doc/architecture.md) describes what exists so far.

Copyright (c) 2016 Maxime Lavoie. Released under the [MIT License](LICENSE).
OpenMind Framework is registered with the Canadian Intellectual Property Office, copyright registration No. 1127739
(2016-01-25).

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e . --group dev
```

## Play

```bash
.venv/bin/openmind-play tictactoe                         # two humans
.venv/bin/openmind-play tictactoe --agent O               # against the MCTS agent
.venv/bin/openmind-play tictactoe/fourinarow --agent O    # 4 in a row, a variant of tic-tac-toe
```

Humans pick legal actions by number; the agent searches with MCTS. The board prints as a grid. Each game's log is saved
in `data/log/play/<domain>/`, such as `data/log/play/tictactoe/fourinarow/`.

## Solve

```bash
.venv/bin/openmind-solve sudoku                              # the puzzle written in the sudoku factory
.venv/bin/openmind-solve sudoku/euler sudoku/top95 sudoku/hardest   # every puzzle of three collections
.venv/bin/openmind-solve sudoku/top95/7                      # one puzzle of a collection
```

Solves domains' constraint problems with the CSP alone. A domain or a single puzzle prints each solution; a collection
prints a summary line per puzzle, then its totals. Each argument's log is saved in `data/log/solve/<argument>/`, for
example `data/log/solve/sudoku/top95/`.

The published sudoku collections are not in the repository. Download them into `data/sudoku/`, which git ignores:

```bash
mkdir -p data/sudoku
curl -o data/sudoku/euler.txt https://projecteuler.net/project/resources/p096_sudoku.txt   # Project Euler, problem 96: 50 puzzles
curl -o data/sudoku/top95.txt https://norvig.com/top95.txt                                 # Peter Norvig: 95 puzzles
curl -o data/sudoku/hardest.txt https://norvig.com/hardest.txt                             # Peter Norvig: 11 puzzles
```

## Distill

```bash
.venv/bin/openmind-distill tictactoe
```

Lets the agent play itself and generates rules from its searches: hypotheses about which moves deserve more or less
exploration, discovered in some games and kept only when a permutation test on held-out games supports them at a
false discovery rate. It prints the rules and the validated hypotheses with their measures. The rule base is saved in
`data/rbs/tictactoe/` and the log in `data/log/distill/tictactoe/`.

## Evaluate

```bash
.venv/bin/openmind-evaluate tictactoe
```

Plays the agent against baselines and compares its choices with perfect play. The report is saved in
`data/evaluation/tictactoe/` and the log in `data/log/evaluate/tictactoe/`. Add `--rules PATH` to evaluate an agent
guided by a distilled rule base: the summary then puts it side by side with an unguided agent on the same positions
(optimal choices, share of the search's visits on optimal moves, mean regret, time per choice), measures the rules
alone, and tests, position by position, whether the rules cut the search's visits on low-value moves and its regret.
`--positions all` measures every position instead of a sample. A domain too large to search for perfect play, such as
`tictactoe/fourinarow`, is measured against long unguided searches with `--reference-iterations N`, or against the
baselines alone with `--positions 0`.

Both `openmind-distill` and `openmind-evaluate` run games and searches in worker processes: `--workers N` sets how many
(half the logical CPUs by default), and the results are the same whatever the number.

## Select

```bash
.venv/bin/openmind-distill tictactoe --explore --held-out-games 20
.venv/bin/openmind-select tictactoe --rules data/rbs/tictactoe/<candidates>.json
```

Generation validates rules one by one; selection keeps the rules that matter for play together. `--explore` generates
many candidate rules, and `openmind-select` removes them one at a time, keeping a removal only when guided searches on
every position play no worse, by a margin on mean regret with a bootstrap bound, than with all the candidates, then
confirms the selection with another seed. The selected rules are saved in `data/rbs/tictactoe/`, the report, updated
after every decision, in `data/selection/tictactoe/`, and the log in `data/log/select/tictactoe/`.

## Distill values

```bash
.venv/bin/openmind-distill-values tictactoe
.venv/bin/openmind-evaluate tictactoe --values data/values/tictactoe/<values>.json
```

Fits value rules: weighted Python terms giving each player's expected payoff in a position, so that a search can value
the positions it reaches instead of playing them to the end, which random play does badly in deep games. Terms are
generated from the domain's rules and the positions of self-play games, and fitted to the payoffs those games ended with
at a sweep of prices on every weight, so the terms that don't pay for themselves drop out; the price whose rules predict
held-out games best is kept. It prints the rules and every price's fit. The value base is saved in
`data/values/tictactoe/` and the log in `data/log/distill-values/tictactoe/`. `openmind-evaluate --values PATH`
searches with them, `--rollout-actions N` playing that many rollout actions before valuing, and also measures the
values alone.

## Problem projects

OpenMind ships without the libraries a problem needs. A problem is programmed in its own project: the project
installs OpenMind, builds its domain with OpenMind's classes, its rules free to import any library, and registers the
domain in its `pyproject.toml`:

```toml
[project.entry-points."openmind.domains"]
chess = "openmind_chess.game.factory.chess_factory:create_chess_domain"
```

Once the project is installed next to OpenMind, OpenMind's commands run its domain: `openmind-play chess`,
`openmind-evaluate chess --rollout-limit 100`. The registered function gets the whole domain name, such as `chess` or
`chess/960`. A project's tests save their logs as OpenMind's do by loading the `openmind.testing.plugin.log_saving`
plugin from their `conftest.py` (see `src/openmind/testing/README.md`).

## Tests

Unit tests sit beside the code they test (`solver.py` → `solver_tests.py`); integration and end-to-end tests live in
`test/`.

```bash
.venv/bin/pytest
```

Each test saves its logs, from DEBUG up, through the `openmind.testing.plugin.log_saving` pytest plugin that
`conftest.py` loads, in `data/log/<test file>/<test name>.log`, for example
`data/log/test/integration/tictactoe_transitions_tests/test_winning_move_sets_payoffs_and_ends_the_game.log`.

`test/integration/sudoku_collections_solve_tests.py` solves every puzzle in `data/sudoku/`, one test per puzzle; it is
skipped when that folder holds no collection.

## Layout

- `src/openmind/`: code, organized by domain
- `data/`: databases, trained models and other data
- `doc/`: documentation
- `test/`: integration and end-to-end tests
