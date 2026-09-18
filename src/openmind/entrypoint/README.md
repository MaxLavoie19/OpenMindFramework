# entrypoint

## Purpose

Every way to run the framework. Entrypoints handle input, output and where logs go; the domains do the work.

## Content

| File | What it is |
|---|---|
| `play.py` | `openmind-play`: play a game in the terminal, as humans or with the agent |
| `evaluate.py` | `openmind-evaluate`: measure how well the agent plays a game and save the report |
| `solve.py` | `openmind-solve`: solve a game's legal actions from where it starts with the CSP alone and print the solutions |
| `distill_values.py` | `openmind-distill-values`: fit position rules on the positions of self-play games, choose a fit on held-out games, and declare them into the knowledge base under a variant of the game |
| `train_values.py` | `openmind-train-values`: play games continuously between arms, remembering every game and every position a decisive game's walk back proved |
| `rollout_options.py` | `add_rollout_limit_options(parser)` and `checked_unfinished_payoff(parser, arguments)`: the rollout limit and its unfinished payoff, which must be given together |
| `dashboard.py` | `openmind-dashboard`: serve a page following a value training while it runs, on the machine it runs on |
| `rerun_call.py` | `openmind-rerun-call`: run again, alone, the call a worker ended on for its memory cap, and show where it holds memory |

## `openmind-rerun-call`

```bash
.venv/bin/openmind-rerun-call data/log/train-values/chess/memory/<YYYY-MM-DD_HH-MM-SS>-worker-<pid>.txt
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 .venv/bin/openmind-rerun-call <diagnosis>   # capped
```

| Option | Default | Meaning |
|---|---|---|
| `--lines N` | `20` | source lines shown, those holding the most memory first |
| `--log-directory DIR` | `data/log/rerun-call` | where logs are saved |

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
| `--report-directory DIR` | `data/training` | where training reports are, under `<domain>/` |
| `--log-directory DIR` | `data/log/train-values` | where training logs are, under `<domain>/` |
| `--syslog PATH` | `/var/log/syslog` | the system log earlyoom writes to; `none` skips it |
| `--dashboard-log-directory DIR` | `data/log/dashboard` | where the dashboard's own log is saved |

Every request takes a fresh snapshot (see `dashboard/README.md`) and serves it as one page: the current round with its
self-play games and deduced moves, whether the training runs and with how many workers, memory and swap, each training
process's memory, every finished round with its games, the latest round's rules, the latest notable log
lines, and earlyoom's latest kills. It runs until stopped; no address can be found without Tailscale and `--host`,
which is rejected.

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

1. Prints the state as the player to act sees it: the cells as a grid under their column numbers, with `.` for an
   empty cell, then one `name = value` line per other variable (`GridTextMapper`, see `world/README.md`).
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

## `openmind-evaluate`

```bash
.venv/bin/openmind-evaluate tictactoe
.venv/bin/openmind-evaluate tictactoe --games 20 --positions 50 --budgets 10,100 --seed 3
.venv/bin/openmind-evaluate tictactoe/fourinarow --positions 0 --games 20   # baselines only
.venv/bin/openmind-evaluate tictactoe --heuristics "tictactoe distilled" --positions all   # the distilled rules against unguided
.venv/bin/openmind-evaluate tictactoe/fourinarow --heuristics "<context>" --positions 50 --reference-iterations 2000   # too large for exact search
```

| Option | Default | Meaning |
|---|---|---|
| `--games N` | `100` | games per baseline series |
| `--iterations N` | `200` | MCTS iterations per move in the baseline series |
| `--positions N` | `100` | positions sampled to measure agreement with perfect play; `all` takes every position; `0` skips agreement and the exact search, for domains too large to search |
| `--budgets LIST` | `10,20,50,100,200,500` | comma-separated iteration budgets for agreement |
| `--seed S` | `1` | random seed |
| `--reference-iterations N` | exact search | stand in for perfect play with unguided searches of N iterations on positions of random games, for domains exact search can't reach; needs a number of `--positions`, not `all` |
| `--heuristics CONTEXT` | none | the context whose rules the evaluated agent plays, such as a round of training: its `move` rules guide it and its `position` rules value the positions its rollouts reach, whichever it holds; the name is recorded in the report |
| `--time-control MINUTES+SECONDS` | no clock | the baseline series play on a clock, the evaluated and the untrained agent both budgeting their moves; the report records the control and each series' wins and losses on time; written as chess writes a time control: `3+2` is 3 minutes and 2 seconds a move; `--iterations` then no longer counts, each move's budget replacing it; a domain without a timeout rule is refused, and so is anything but minutes+seconds or a base of 0 |
| `--expected-steps N` | `30` | with a clock: steps a player expects to be left at any point of a game, the plain time budget estimator sharing the time left between them (see `timing/README.md`); below 1 is refused |
| `--time-reserve X` | `0.05` | with a clock: the share of a player's base time kept in reserve; below it the player plays only random moves, so its clock never runs out; from 0 to below 1 |
| `--selection ucb1\|puct` | `ucb1` | how a tried node picks the action to follow: `ucb1` tries every legal action once first, `puct` follows Q + c · P · √N / (1 + n) with a prior (see `mcts/README.md`) |
| `--prior uniform\|rater\|value` | `uniform` | the prior PUCT follows: every action alike, the agent's rules' ratings, or its value rules' values of each action's outcomes; `rater` and `value` need `--heuristics`; the untrained agent always follows `uniform` |
| `--puct-exploration X` | `1.5` | PUCT's exploration weight c; 0 or more |
| `--prior-temperature X` | `0.1` | the softmax temperature turning ratings or values into a prior, lower following the best more closely; above 0 |
| `--knowledge DIR` | `data/knowledge` | where the knowledge base remembers every game with its models, under `<domain>/` (see `agent/README.md`, `GameMemory`) |
| `--workers N` | half the logical CPUs | worker processes baseline games, reference searches and the positions searched at each budget run in; the report is the same whatever the number, apart from seconds per choice, and every worker holds its own caches, so memory grows with it |
| `--rollouts MODE` | `guided` | with move rules: `guided` rollouts follow the rules' ratings; `unguided` rollouts pick uniformly and only the search tree's nodes are rated, which is much cheaper with rules reading lookahead such as `wins()` |
| `--rollout-actions N` | `0` | with position rules: rollout actions played before valuing a position; 0 values the search's new position itself |
| `--rollout-limit N` | no limit | actions a rollout plays at most, for every agent the evaluation builds (evaluated, unguided, untrained), before every player gets the unfinished payoff; reference searches play to the end |
| `--unfinished-payoff X` | none | each player's payoff for a rollout stopped at the limit; needed with `--rollout-limit`, and rejected without it |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/evaluate` | where logs are saved |
| `--report-directory DIR` | `data/evaluation` | where reports are saved |

Runs the measures described in `evaluation/README.md` and prints the report's JSON, then its summary: the baseline
results, then a table of optimal choices, visit share on optimal actions, mean regret and seconds per choice at each
budget. With move rules, the table shows the guided and the unguided agent side by side (`guided / unguided`) on the
same positions and is followed by the rules alone and the paired tests of guided against unguided (see
`evaluation/README.md`):

```
Guided by <context> against unguided, on the same 100 positions (every action optimal in <n>):
iterations  optimal  visits on optimal    mean regret   seconds per choice
        10  <g> / <u>  ...
Rules alone: ratings separate actions in <n> of 100 positions; a top-rated action is optimal in <expected> of 100; mean regret <regret>
Guided against unguided, paired by position (guided minus unguided; Wilcoxon and McNemar p-values):
iterations  low-value visits  p  regret  p  optimal only guided / unguided  p
        10            -0.052  ...
```

With `--rollouts unguided`, the heading reads `Guided by <context> with unguided rollouts against unguided`. With
position rules, the heading reads `Valued by <context> against unguided`, or `Guided by <context> and valued by
<context>` with both, adding `after N rollout actions` with `--rollout-actions N`, and the values alone follow the rules
alone:

```
Values alone: valued <n> of 100 positions, mean absolute error <error>; one step ahead, a top-valued action is optimal in <expected> of 100; mean regret <regret>
```

With `--reference-iterations N`, the heading names the reference: `(reference: N-iteration unguided searches; every
action optimal in <n>)`; with `--rollout-limit N`, it adds `rollout limit N, unfinished payoff X; ` before `every
action optimal`.

It saves the report as
`<report directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, ending with `INFO Saved report <path>` from logger
`openmind.entrypoint.evaluate`.

With `--heuristics`, the log also has `INFO Evaluating with the move rules of <context>` and `INFO Evaluating with
the position rules of <context>`, for whichever kinds it holds. With `--rollout-limit`, the log of `openmind-distill-values` has `INFO Self-play
rollouts stop after <n> actions, every player getting <payoff>`. Every log starts with `INFO Running games and searches
in <n> worker processes`.

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
| `--seconds X` | `3600.0` | seconds the inference engine's expression search runs (see `inference/README.md`) |
| `--memory X` | half the machine's memory | GB the expression search's process holds at most, measured; its workers each hold an even share; this process's caches are cleared above it |
| `--candidates N` | no limit | candidates the expression search tries at most; a search limited by candidates, unlike one limited by time, gives the same rules on any machine |
| `--training-time-control MINUTES+SECONDS` | no clock | self-play games play on a clock, every agent budgeting its moves; written as chess writes a time control: `3+2` is 3 minutes and 2 seconds a move; `--iterations` then no longer counts, each move's budget replacing it; a domain without a timeout rule is refused, and so is anything but minutes+seconds or a base of 0 |
| `--expected-steps N` | `30` | with a clock: steps a player expects to be left at any point of a game, the plain time budget estimator sharing the time left between them (see `timing/README.md`); below 1 is refused |
| `--time-reserve X` | `0.05` | with a clock: the share of a player's base time kept in reserve; below it the player plays only random moves, so its clock never runs out; from 0 to below 1 |
| `--selection ucb1\|puct` | `ucb1` | how a tried node picks the action to follow: `ucb1` tries every legal action once first, `puct` follows Q + c · P · √N / (1 + n) with a prior (see `mcts/README.md`) |
| `--prior uniform\|rater\|value` | `uniform` | the prior PUCT follows: every action alike, the agent's rules' ratings, or its value rules' values of each action's outcomes; only `uniform` here, the self-play agent having no rules |
| `--puct-exploration X` | `1.5` | PUCT's exploration weight c; 0 or more |
| `--prior-temperature X` | `0.1` | the softmax temperature turning ratings or values into a prior, lower following the best more closely; above 0 |
| `--knowledge DIR` | `data/knowledge` | where the knowledge base remembers every game with its models, under `<domain>/` (see `agent/README.md`, `GameMemory`) |
| `--prices LIST` | `0.1,0.03,0.01,0.003,0.001` | comma-separated L1 prices swept |
| `--max-steps N` | `1000` | steps a fit takes at most |
| `--tolerance X` | `1e-06` | weight change below which a fit has settled |
| `--workers N` | half the logical CPUs | worker processes self-play games and term evaluations run in; the value rules are the same whatever the number |
| `--worker-memory X` | `--memory` shared between the workers | GB each worker process holds at most: over it a worker clears its caches, a worker that stays over is ended with a diagnosis in `<log directory>/<domain>/memory/` and its call runs again in a fresh worker, and a call over it twice is dropped (a game, a pondered position) or stops the expression search (see `parallel/README.md`) |
| `--rollout-limit N` | no limit | actions a self-play rollout plays at most before every player gets the unfinished payoff; domains whose random games run long, such as chess, need one |
| `--unfinished-payoff X` | none | each player's payoff for a rollout stopped at the limit; needed with `--rollout-limit`, and rejected without it |
| `--deduction-plies N` | `0` | actions ahead a deduction of one position looks at most: self-play agents deduce the positions their rules have no clue about; `0` never deduces |
| `--deduction-seconds X` | `10.0` | seconds a deduction of one position runs at most |
| `--highest-payoff X` | `1.0` | the highest payoff a player can get: a move proven to reach it needs no comparison with moves not proven yet |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log |
| `--log-directory DIR` | `data/log/distill-values` | where logs are saved |
| `--context NAME` | `<domain> distilled` | the context the fitted position rules are declared under, a variant of the game carrying its rules |

Runs the value distillation described in `training/README.md` and `rbs/README.md`, and prints:

```
<weight> × <rule>, one line each, such as +0.42 × wins(me), the constant first
Position rules: <n> of <m> candidate terms, chosen at price <price>; declared under <context>
price  terms kept  steps  settled  training loss  held-out loss
  0.1          <k>    <s>      yes       <loss>          <loss>
Rows: <training> for training, <held out> held out, valued at the <target> target; mean absolute error on held-out rows: <error>
Declared under <context>
```

`--deduction-seconds` of 0 or less with `--deduction-plies`, `--memory` or `--worker-memory` of 0 or less, and
`--rollout-limit` and `--unfinished-payoff` given one without the other are rejected.

It declares the fitted position rules into the knowledge base, under `--context`, and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`: it starts with `INFO Running self-play and term evaluations in
<n> worker processes, each holding at most <bytes> bytes; memory diagnoses in <directory>` and `INFO Valuing positions at the <target> target; prices <prices>; searching expressions for <seconds> seconds within
<bytes> bytes, trying any number of|at most <n> candidates`, has the search's generations (see `inference/README.md`) and the generator's fits (see
`rbs/README.md`), and ends with `INFO Declared the position rules under <context>` from logger
`openmind.entrypoint.distill_values`.

## `openmind-train-values`

```bash
.venv/bin/openmind-train-values tictactoe --games 20
.venv/bin/openmind-train-values chess --training-time-control 5+0 --deduction-plies 3 --deduction-seconds 2 --ponder-endings 20
```

Plays games continuously between arms (see "How continuous training works" in `training/README.md`), remembering every
game and every position a decisive game's walk back proved. Nothing is learned from the games yet.

| Option | Default | Meaning |
|---|---|---|
| `--games N` | until stopped | games to play, then stop |
| `--iterations N` | `100` | MCTS iterations per move without a clock; on a clock each move's budget replaces them |
| `--training-time-control MINUTES+SECONDS` | no clock | games play on a clock, every agent budgeting its moves; written as chess writes a time control: `3+2` is 3 minutes and 2 seconds a move; a domain without a timeout rule is rejected |
| `--expected-steps N` | `30` | with a clock: steps a player expects to be left at any point of a game, the plain time budget estimator sharing the time left between them |
| `--time-reserve X` | `0.05` | with a clock: the share of a player's base time kept in reserve; below it the player plays only random moves, so its clock never runs out |
| `--selection ucb1\|puct` | `ucb1` | how a tried node picks the action to follow: `ucb1` tries every legal action once first, `puct` follows Q + c · P · √N / (1 + n) over every legal action |
| `--prior uniform\|rater\|value` | `uniform` | the prior PUCT follows: every action alike, or each arm's own value rules' values of each action's outcomes; `rater` is rejected, training having no rules that rate actions |
| `--puct-exploration X` | `1.5` | PUCT's exploration weight c; 0 or more |
| `--prior-temperature X` | `0.1` | the softmax temperature turning values into a prior, lower following the best more closely; above 0 |
| `--knowledge DIR` | `data/knowledge` | where the knowledge base remembers every game with its models and every proof, under `<domain>/` |
| `--seed S` | `1` | random seed |
| `--memory X` | half the machine's memory | GB this process holds before clearing its caches, shared evenly between the workers unless `--worker-memory` says otherwise |
| `--rollout-actions N` | `10` | rollout actions played before a position is valued with value rules |
| `--rollout-limit N` | no limit | actions a rollout plays at most before every player gets the unfinished payoff |
| `--unfinished-payoff X` | none | each player's payoff for a rollout stopped at the limit; needed with `--rollout-limit`, and rejected without it |
| `--deduction-plies N`, `--deduction-seconds X`, `--highest-payoff X` | as `openmind-distill-values` | the deduction every agent falls back on when its rules have no clue, and which walks decisive games back |
| `--ponder-endings N` | `0` | positions of each decisive game deduced at most, walking back from its end until one isn't proven; needs `--deduction-plies` |
| `--workers N` | half the logical CPUs | worker processes games are played and studied in |
| `--worker-memory X` | `--memory` shared between the workers | as `openmind-distill-values` |
| `--arm-exploration X` | `1.414...` (√2) | UCB1's exploration weight when a worker taking a game chooses which arms play it; below 0 is rejected |
| `--arm-library PATH` | none: games played without position rules | the context each arm plays, a variant of the game carrying its own position rules; a library for another game is rejected; an arm written under the older `signal` key reads the same |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log |
| `--log-directory DIR` | `data/log/train-values` | where logs are saved |

Games are numbered after those the knowledge base already remembers. The log,
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, starts with `INFO Playing in <n> worker processes, each holding at
most <bytes> bytes; memory diagnoses in <directory>; arms from <path or no arm library>` from logger
`openmind.entrypoint.train_values`, then has the games' lines (see `training/README.md`).

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
  `test/end_to_end/distill_values_tictactoe_tests.py`, `test/end_to_end/train_values_tictactoe_tests.py`,
  `test/end_to_end/solve_sudoku_tests.py`.
