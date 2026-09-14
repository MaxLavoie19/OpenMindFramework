# entrypoint

## Purpose

Every way to run the framework. Entrypoints handle input, output and where logs go; the domains do the work.

## Content

| File | What it is |
|---|---|
| `play.py` | `openmind-play`: play a domain within the agent in the terminal, as humans or with the agent |
| `evaluate.py` | `openmind-evaluate`: measure how well the agent plays a domain and save the report |
| `distill.py` | `openmind-distill`: generate a rule base from self-play, validate it on held-out games, and save it |
| `solve.py` | `openmind-solve`: solve a domain's constraint problem with the CSP alone and print the solutions |
| `select.py` | `openmind-select`: select the smallest set of a rule base's rules that plays no worse than all of them |
| `distill_values.py` | `openmind-distill-values`: fit value rules on the positions of self-play games, choose a fit on held-out games, and save the value base |

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
| `--log-level LEVEL` | `INFO` | lowest level saved in the game log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/play` | where game logs are saved |

1. Prints the state: the cells as a grid under their column numbers, with `.` for an empty cell, then one
   `name = value` line per other variable (`GridTextMapper`, see `world/README.md`).
2. On a human's turn, lists the legal actions by number and reads the number of the action to perform. Anything else
   asks again; end of input (Ctrl+D) ends the session.
3. On an agent's turn, the agent searches and the CLI prints `<player> chose <action>`.
4. The predictor gives the outcome distribution; when there are several outcomes, one is drawn by its probability.
5. When no action is legal, prints the final state, payoffs included.

Domains are created by name through `agent/factory/domain_factory.py`: `tictactoe`, its variants
`tictactoe/fourinarow` and `tictactoe/gomoku`, and `sudoku`. An `--agent` player the domain doesn't have is rejected.

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

## `openmind-evaluate`

```bash
.venv/bin/openmind-evaluate tictactoe
.venv/bin/openmind-evaluate tictactoe --games 20 --positions 50 --budgets 10,100 --seed 3
.venv/bin/openmind-evaluate tictactoe/fourinarow --positions 0 --games 20   # baselines only
.venv/bin/openmind-evaluate tictactoe --rules data/rbs/tictactoe/<rule base>.json --positions all   # guided against unguided
.venv/bin/openmind-evaluate tictactoe/fourinarow --rules <rule base>.json --positions 50 --reference-iterations 2000   # too large for exact search
.venv/bin/openmind-evaluate tictactoe --values data/values/tictactoe/<value base>.json --positions all   # valuing positions against unguided
```

| Option | Default | Meaning |
|---|---|---|
| `--games N` | `100` | games per baseline series |
| `--iterations N` | `200` | MCTS iterations per move in the baseline series |
| `--positions N` | `100` | positions sampled to measure agreement with perfect play; `all` takes every position; `0` skips agreement and the exact search, for domains too large to search |
| `--budgets LIST` | `10,20,50,100,200,500` | comma-separated iteration budgets for agreement |
| `--seed S` | `1` | random seed |
| `--reference-iterations N` | exact search | stand in for perfect play with unguided searches of N iterations on positions of random games, for domains exact search can't reach; needs a number of `--positions`, not `all` |
| `--rules PATH` | none | rule base guiding the evaluated agent; its path is recorded in the report |
| `--workers N` | half the logical CPUs | worker processes baseline games, reference searches and the positions searched at each budget run in; the report is the same whatever the number, apart from seconds per choice, and every worker holds its own caches, so memory grows with it |
| `--rollouts MODE` | `guided` | with `--rules`: `guided` rollouts follow the rules' ratings; `unguided` rollouts pick uniformly and only the search tree's nodes are rated, which is much cheaper with rules reading lookahead such as `wins()` |
| `--values PATH` | none | value base valuing the positions the evaluated agent's rollouts reach; its path is recorded in the report |
| `--rollout-actions N` | `0` | with `--values`: rollout actions played before valuing a position; 0 values the search's new position itself |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/evaluate` | where logs are saved |
| `--report-directory DIR` | `data/evaluation` | where reports are saved |

Runs the measures described in `evaluation/README.md` and prints the report's JSON, then its summary: the baseline
results, then a table of optimal choices, visit share on optimal actions, mean regret and seconds per choice at each
budget. With `--rules`, the table shows the guided and the unguided agent side by side (`guided / unguided`) on the
same positions and is followed by the rules alone and the paired tests of guided against unguided (see
`evaluation/README.md`):

```
Guided by data/rbs/tictactoe/<rule base>.json against unguided, on the same 100 positions (every action optimal in <n>):
iterations  optimal  visits on optimal    mean regret   seconds per choice
        10  <g> / <u>  ...
Rules alone: ratings separate actions in <n> of 100 positions; a top-rated action is optimal in <expected> of 100; mean regret <regret>
Guided against unguided, paired by position (guided minus unguided; Wilcoxon and McNemar p-values):
iterations  low-value visits  p  regret  p  optimal only guided / unguided  p
        10            -0.052  ...
```

With `--rollouts unguided`, the heading reads `Guided by <rule base> with unguided rollouts against unguided`. With
`--values`, the heading reads `Valued by <value base> against unguided`, or `Guided by <rule base> and valued by <value
base>` with both, adding `after N rollout actions` with `--rollout-actions N`, and the values alone follow the rules
alone:

```
Values alone: valued <n> of 100 positions, mean absolute error <error>; one step ahead, a top-valued action is optimal in <expected> of 100; mean regret <regret>
```

With `--reference-iterations N`, the heading names the reference: `(reference: N-iteration unguided searches; every
action optimal in <n>)`.

It saves the report as
`<report directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, ending with `INFO Saved report <path>` from logger
`openmind.entrypoint.evaluate`.

With `--rules`, the log also has `INFO Evaluating with rules <path>`, and with `--values`,
`INFO Evaluating with values <path>`. Every log starts with
`INFO Running games and searches in <n> worker processes`, and `openmind-distill`'s with
`INFO Running self-play and rule generation in <n> worker processes`.

## `openmind-distill`

```bash
.venv/bin/openmind-distill tictactoe
.venv/bin/openmind-distill tictactoe --games 40 --held-out-games 40 --iterations 500 --max-conditions 3
```

| Option | Default | Meaning |
|---|---|---|
| `--games N` | `20` | self-play games to discover rules in |
| `--held-out-games N` | `5` | self-play games to validate and measure rules on; validation needs enough of them to reach the false discovery rate |
| `--iterations N` | `200` | MCTS iterations per self-play move |
| `--seed S` | `1` | random seed, also drawing the validation permutations |
| `--min-visits N` | `5` | visits an action needs in a state to count |
| `--max-conditions N` | `2` | conditions per rule at most |
| `--min-rule-visits N` | `50` | visits the actions a rule matches need |
| `--min-gain X` | `0.05` | difference in advantage a hypothesis needs in discovery |
| `--confidence X` | `0.95` | confidence of the payoff bound that marks priority rules |
| `--beam-width N` | `20` | hypotheses of each size kept and extended |
| `--max-offset N` | `2` | largest index offset `near()` reads from the variable an action sets |
| `--solo-limit N` | `2` | own moves `solo_distance()` looks ahead; `0` leaves it out |
| `--patterns N` | `200` | winning moves probed for goal patterns |
| `--false-discovery-rate X` | `0.05` | false discovery rate hypotheses are kept at |
| `--permutations N` | `10000` | permutations of each validation test |
| `--workers N` | half the logical CPUs | worker processes self-play games and rule condition checks run in; the rules are the same whatever the number |
| `--explore` | off | generate many candidates for `openmind-select`: beam width 60, up to 3 conditions, min gain 0.02, false discovery rate 0.2 and no coverage, each unless given explicitly |
| `--coverage`, `--no-coverage` | on, off with `--explore` | whether validated rules a simpler rule covers are left out |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/distill` | where logs are saved |
| `--rules-directory DIR` | `data/rbs` | where rule bases are saved |

Runs the distillation described in `training/README.md` and `rbs/README.md`. It prints every rule as text, then:

```
Rules: <n>, conditions per rule on average: <mean>
Goal patterns: <n>; hypotheses: <m> tested, <k> validated at a false discovery rate of <q>; <c> covered by a simpler rule
  <validated hypothesis as text>, one line each
  <covered rule as text>: covered by <covering rule as text>, one line each
Samples: <training> for training, <held out> held out; rating error on held-out samples: <error>
Saved rules <path>
```

Every hypothesis, rejected ones included, is in the log at `--log-level DEBUG`. It saves the rule base as
`<rules directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, ending with `INFO Saved rules <path>` from logger
`openmind.entrypoint.distill`.

## `openmind-select`

```bash
.venv/bin/openmind-distill tictactoe --explore --held-out-games 20          # many candidate rules
.venv/bin/openmind-select tictactoe --rules data/rbs/tictactoe/<candidates>.json
.venv/bin/openmind-select tictactoe/fourinarow --rules <candidates>.json --positions 500 --reference-iterations 2000 --iterations 50 --rollouts unguided --max-hours 4
```

| Option | Default | Meaning |
|---|---|---|
| `--rules PATH` | required | candidate rule base |
| `--positions N` | `all` | positions rule sets are compared on; `all` needs exact search |
| `--reference-iterations N` | exact search | values from unguided searches of N iterations on positions of random games, for domains exact search can't reach; needs a number of `--positions` |
| `--iterations N` | `10` | MCTS iterations of the guided searches compared |
| `--rollouts MODE` | `guided` | whether the guided searches' rollouts follow the ratings |
| `--margin X` | `0.005` | largest rise in mean regret a removed rule may cause |
| `--confidence X` | `0.95` | confidence of the one-sided bound on that rise |
| `--resamples N` | `10000` | bootstrap resamples per comparison |
| `--seed S` | `1` | random seed; the confirmation uses S + 1 |
| `--max-hours H` | no limit | stop trying removals after H hours, then confirm what is selected |
| `--workers N` | half the logical CPUs | worker processes the searches run in |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log |
| `--log-directory DIR` | `data/log/select` | where logs are saved |
| `--rules-directory DIR` | `data/rbs` | where the selected rule base is saved |
| `--report-directory DIR` | `data/selection` | where reports are saved |

Runs the selection described in `training/README.md`. The report is written to
`<report directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` after every decision, so a long selection can be followed.
At the end it saves the selected rules as `<rules directory>/<domain>/<same time>.json` and prints:

```
Selected <n> of <m> candidate rules in <p> passes, complete: <r> removed, <f> of them without a search
At 10 iterations on 4520 positions: all rules <o> optimal choices, mean regret <x>; selected rules <o> optimal choices, mean regret <x>
Confirmation with seed 2 on 4520 positions: regret <d>, upper bound <b>, below the margin 0.005
Selected rules:
  <rule as text>, one line each
Saved selected rules <path>
Saved selection report <path>
```

The log, `<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, starts with `INFO Selecting from <m> rules in <path>,
searching in <n> worker processes`, has the selector's decisions (see `training/README.md`), and ends with
`INFO Saved selected rules <path>` and `INFO Saved selection report <path>` from logger `openmind.entrypoint.select`.

## `openmind-distill-values`

```bash
.venv/bin/openmind-distill-values tictactoe
.venv/bin/openmind-distill-values tictactoe/fourinarow --games 200 --held-out-games 50 --target search
```

| Option | Default | Meaning |
|---|---|---|
| `--games N` | `100` | self-play games to fit value rules on |
| `--held-out-games N` | `25` | self-play games to choose a fit and measure it on |
| `--iterations N` | `200` | MCTS iterations per self-play move |
| `--seed S` | `1` | random seed |
| `--target TARGET` | `outcome` | what a position is valued at: `outcome`, the game's final payoff for each player, or `search`, the search's mean payoff for the player to act |
| `--pair-pool N` | `20` | single terms, the most correlated with the payoffs, multiplied in pairs |
| `--cuts N` | `6` | thresholds a quantity is cut at, at most |
| `--solo-limit N` | `2` | own actions `solo_distance()` looks ahead; `0` leaves it out |
| `--prices LIST` | `0.1,0.03,0.01,0.003,0.001` | comma-separated L1 prices swept |
| `--max-steps N` | `1000` | steps a fit takes at most |
| `--tolerance X` | `1e-06` | weight change below which a fit has settled |
| `--workers N` | half the logical CPUs | worker processes self-play games and term evaluations run in; the value rules are the same whatever the number |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log |
| `--log-directory DIR` | `data/log/distill-values` | where logs are saved |
| `--values-directory DIR` | `data/values` | where value bases are saved |

Runs the value distillation described in `training/README.md` and `rbs/README.md`, and prints:

```
bias <bias>
<value rule as text>, one line each, such as +0.42 × wins(me)
Value rules: <n> of <m> candidate terms, chosen at price <price>; payoffs from <low> to <high>
price  terms kept  steps  settled  training loss  held-out loss
  0.1          <k>    <s>      yes       <loss>          <loss>
Rows: <training> for training, <held out> held out, valued at the <target> target; mean absolute error on held-out rows: <error>
Saved values <path>
```

It saves the value base as `<values directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`: it starts with `INFO Running self-play and term evaluations in
<n> worker processes` and `INFO Valuing positions at the <target> target; pair pool <n>, up to <c> cuts, solo limit <s>,
prices <prices>`, has the generator's fits (see `rbs/README.md`), and ends with `INFO Saved values <path>` from logger
`openmind.entrypoint.distill_values`.

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

- End-to-end tests: `test/end_to_end/play_tictactoe_tests.py`, `test/end_to_end/play_fourinarow_tests.py`,
  `test/end_to_end/evaluate_tictactoe_tests.py`, `test/end_to_end/evaluate_fourinarow_tests.py`,
  `test/end_to_end/distill_tictactoe_tests.py`, `test/end_to_end/select_tictactoe_tests.py`,
  `test/end_to_end/distill_values_tictactoe_tests.py`, `test/end_to_end/solve_sudoku_tests.py`.
