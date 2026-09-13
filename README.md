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

## Tests

Unit tests sit beside the code they test (`solver.py` → `solver_tests.py`); integration and end-to-end tests live in
`test/`.

```bash
.venv/bin/pytest
.venv/bin/pytest test/integration -o log_cli=true --log-cli-level=DEBUG   # with logs
```

## Layout

- `src/openmind/`: code, organized by domain
- `data/`: databases, trained models and other data
- `doc/`: documentation
- `test/`: integration and end-to-end tests
