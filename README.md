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

Lets the agent play itself, induces rules from its searches, and prints them with their measures. The rule base is
saved in `data/rbs/tictactoe/` and the log in `data/log/distill/tictactoe/`.

## Evaluate

```bash
.venv/bin/openmind-evaluate tictactoe
```

Plays the agent against baselines and compares its choices with perfect play. The report is saved in
`data/evaluation/tictactoe/` and the log in `data/log/evaluate/tictactoe/`. Add `--rules PATH` to evaluate an agent
guided by a distilled rule base. A domain too large to search for perfect play, such as `tictactoe/fourinarow`, is
evaluated against the baselines alone with `--positions 0`.

## Tests

Unit tests sit beside the code they test (`solver.py` → `solver_tests.py`); integration and end-to-end tests live in
`test/`.

```bash
.venv/bin/pytest
```

Each test saves its logs, from DEBUG up, in `data/log/<test file>/<test name>.log`, for example
`data/log/test/integration/tictactoe_transitions_tests/test_winning_move_sets_payoffs_and_ends_the_game.log`.

`test/integration/sudoku_collections_solve_tests.py` solves every puzzle in `data/sudoku/`, one test per puzzle; it is
skipped when that folder holds no collection.

## Layout

- `src/openmind/`: code, organized by domain
- `data/`: databases, trained models and other data
- `doc/`: documentation
- `test/`: integration and end-to-end tests
