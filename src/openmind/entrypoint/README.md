# entrypoint

## Purpose

Every way to run the framework. Entrypoints handle input, output and where logs go; the domains do the work.

## Content

| File | What it is |
|---|---|
| `play.py` | `openmind-play`: play a domain within the agent in the terminal, as humans or with the agent |
| `evaluate.py` | `openmind-evaluate`: measure how well the agent plays a domain and save the report |
| `distill.py` | `openmind-distill`: distill a rule base from self-play and save it |
| `solve.py` | `openmind-solve`: solve a domain's constraint problem with the CSP alone and print the solutions |

## `openmind-play`

```bash
.venv/bin/openmind-play tictactoe                                 # two humans
.venv/bin/openmind-play tictactoe --agent O                       # human X against the agent
.venv/bin/openmind-play tictactoe --agent X --agent O --seed 7    # the agent against itself
```

| Option | Default | Meaning |
|---|---|---|
| `--agent PLAYER` | none | a player the agent controls; repeat for several |
| `--iterations N` | `1000` | MCTS iterations per agent move |
| `--seed S` | unseeded | random seed for the agent's search |
| `--log-level LEVEL` | `INFO` | lowest level saved in the game log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/play` | where game logs are saved |

1. Prints the state, one `name = value` line per variable.
2. On a human's turn, lists the legal actions by number and reads the number of the action to perform. Anything else
   asks again; end of input (Ctrl+D) ends the session.
3. On an agent's turn, the agent searches and the CLI prints `<player> chose <action>`.
4. The predictor gives the outcome distribution; when there are several outcomes, one is drawn by its probability.
5. When no action is legal, prints the final state, payoffs included.

Domains are created by name through `agent/factory/domain_factory.py`; an `--agent` player the domain doesn't have is
rejected.

Each session writes `<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, from `--log-level` up, as
`LEVEL logger message` lines. At `INFO`: the agent's search results (see `mcts/README.md`), plus, from logger
`openmind.entrypoint.play`:

- `INFO Playing <domain>`
- `INFO Chose <action>`
- `INFO No legal action left: game over`
- `INFO Input ended before the game was over`

At `DEBUG`, every solver candidate, predictor effect and search iteration is added, including those inside the
agent's rollouts.

## `openmind-evaluate`

```bash
.venv/bin/openmind-evaluate tictactoe
.venv/bin/openmind-evaluate tictactoe --games 20 --positions 50 --budgets 10,100 --seed 3
```

| Option | Default | Meaning |
|---|---|---|
| `--games N` | `100` | games per baseline series |
| `--iterations N` | `200` | MCTS iterations per move in the baseline series |
| `--positions N` | `100` | positions sampled to measure agreement with perfect play |
| `--budgets LIST` | `10,20,50,100,200,500` | comma-separated iteration budgets for agreement |
| `--seed S` | `1` | random seed |
| `--rules PATH` | none | rule base guiding the evaluated agent; its path is recorded in the report |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/evaluate` | where logs are saved |
| `--report-directory DIR` | `data/evaluation` | where reports are saved |

Runs the measures described in `evaluation/README.md`, prints the report's JSON, saves it as
`<report directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, ending with `INFO Saved report <path>` from logger
`openmind.entrypoint.evaluate`.

With `--rules`, the log also has `INFO Evaluating with rules <path>`.

## `openmind-distill`

```bash
.venv/bin/openmind-distill tictactoe
.venv/bin/openmind-distill tictactoe --games 40 --iterations 500 --max-conditions 3
```

| Option | Default | Meaning |
|---|---|---|
| `--games N` | `20` | self-play games to learn from |
| `--held-out-games N` | `5` | self-play games to measure the rules on |
| `--iterations N` | `200` | MCTS iterations per self-play move |
| `--seed S` | `1` | random seed |
| `--min-visits N` | `5` | visits a sample needs to count |
| `--max-conditions N` | `2` | conditions per rule at most |
| `--min-rule-visits N` | `50` | visits a rule needs |
| `--min-gain X` | `0.05` | change in expected value a condition needs |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/distill` | where logs are saved |
| `--rules-directory DIR` | `data/rbs` | where rule bases are saved |

Runs the distillation described in `training/README.md`. It prints every rule as text, then the number of rules and
conditions per rule, the sample counts and the rating error on held-out samples. It saves the rule base as
`<rules directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, ending with `INFO Saved rules <path>` from logger
`openmind.entrypoint.distill`.

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
| `--puzzle-directory DIR` | `data/sudoku` | where sudoku collections are read from, one `<collection>.txt` file each (the root README has the downloads) |

Every DOMAIN is resolved before anything is solved. An unknown domain raises `ValueError`; a collection missing from
the puzzle directory is rejected with the collections found; a puzzle number outside its collection is rejected with
the collection's range.

One solver solves every DOMAIN in turn from its initial state (see `csp/README.md`):

- **A domain or a single puzzle:** for each solution, the predictor applies the action and the CLI prints
  `Solution <n> (probability <p>):` followed by the resulting state, one `name = value` line per variable. A summary
  line follows.
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

- End-to-end tests: `test/end_to_end/play_tictactoe_tests.py`, `test/end_to_end/evaluate_tictactoe_tests.py`,
  `test/end_to_end/distill_tictactoe_tests.py`, `test/end_to_end/solve_sudoku_tests.py`.
