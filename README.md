# OpenMind Framework

An AI framework in which an agent works within a domain by searching with Monte-Carlo Tree Search (MCTS). A constraint
satisfaction problem generates the possible actions and their parameters, a predictor gives the probability of each
outcome, and a strategy rule-based system distills best play, weighted by expected value. The first domain being built
is tic-tac-toe. The framework grows in small increments; [doc/architecture.md](doc/architecture.md) describes what
exists so far.

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
.venv/bin/openmind-play tictactoe              # two humans
.venv/bin/openmind-play tictactoe --agent O    # against the MCTS agent
```

Humans pick legal actions by number; the agent searches with MCTS. Each game's log is saved in
`data/log/play/tictactoe/`.

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
guided by a distilled rule base.

## Tests

Unit tests sit beside the code they test (`solver.py` → `solver_tests.py`); integration and end-to-end tests live in
`test/`.

```bash
.venv/bin/pytest
```

Each test saves its logs, from DEBUG up, in `data/log/<test file>/<test name>.log`, for example
`data/log/test/integration/tictactoe_transitions_tests/test_winning_move_sets_payoffs_and_ends_the_game.log`.

## Layout

- `src/openmind/`: code, organized by domain
- `data/`: databases, trained models and other data
- `doc/`: documentation
- `test/`: integration and end-to-end tests
