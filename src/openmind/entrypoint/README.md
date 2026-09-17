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
| `train_values.py` | `openmind-train-values`: train value rules continuously, learning from every game as it ends, a rule search after every decisive game, the signal library saved after every game |
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
process's memory, every finished round with its games and pondering, the latest round's rules, the latest notable log
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
| `--unfinished-payoff X` | `0.5` | with `--rollout-limit`: each player's payoff for a rollout stopped at the limit, a draw in games paying 1, 0.5 and 0 |
| `--time-control MINUTES+SECONDS` | no clock | every player, human or agent, plays on a clock, a human's thinking at the prompt included; both clocks are printed before each move, and a player whose time runs out loses as the domain's timeout rule says, the game ending with `<player>'s time ran out`; written as chess writes a time control: `3+2` is 3 minutes and 2 seconds a move; `--iterations` then caps each move; a domain without a timeout rule is refused, and so is anything but minutes+seconds or a base of 0 |
| `--expected-steps N` | `30` | with a clock: steps a player expects to be left at any point of a game, the plain time budget estimator sharing the time left between them (see `timing/README.md`); below 1 is refused |
| `--selection ucb1\|puct` | `ucb1` | how a tried node picks the action to follow: `ucb1` tries every legal action once first, `puct` follows Q + c · P · √N / (1 + n) with a prior (see `mcts/README.md`) |
| `--prior uniform\|rater\|value` | `uniform` | the prior PUCT follows: every action alike, the agent's rules' ratings, or its value rules' values of each action's outcomes; only `uniform` here, the agent having no rules when playing |
| `--puct-exploration X` | `1.5` | PUCT's exploration weight c; 0 or more |
| `--prior-temperature X` | `0.1` | the softmax temperature turning ratings or values into a prior, lower following the best more closely; above 0 |
| `--log-level LEVEL` | `INFO` | lowest level saved in the game log: `DEBUG`, `INFO` or `WARNING` |
| `--log-directory DIR` | `data/log/play` | where game logs are saved |

1. Prints the state as the player to act sees it: the cells as a grid under their column numbers, with `.` for an
   empty cell, then one `name = value` line per other variable (`GridTextMapper`, see `world/README.md`). In a domain
   with an observation, such as the prisoner's dilemma, a variable hidden from that player shows `<hidden>`, and the
   agent searches from the same view.
2. On a human's turn, lists the legal actions by number and reads the number of the action to perform. Anything else
   asks again; end of input (Ctrl+D) ends the session.
3. On an agent's turn, the agent searches and the CLI prints `<player> chose <action>`.
4. The predictor gives the outcome distribution; when there are several outcomes, one is drawn by its probability.
5. When no action is legal, prints the final state, payoffs included.

Domains are created by name through `agent/factory/domain_factory.py`: `tictactoe`, its variants
`tictactoe/fourinarow` and `tictactoe/gomoku`, `sudoku`, `prisonersdilemma` and its variant
`prisonersdilemma/uncertain`, and any domain an installed project registers (see
`agent/README.md`). An `--agent` player the domain doesn't have is rejected.

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
| `--time-control MINUTES+SECONDS` | no clock | the baseline series play on a clock, the evaluated and the untrained agent both budgeting their moves; the report records the control and each series' wins and losses on time; written as chess writes a time control: `3+2` is 3 minutes and 2 seconds a move; `--iterations` then caps each move; a domain without a timeout rule is refused, and so is anything but minutes+seconds or a base of 0 |
| `--expected-steps N` | `30` | with a clock: steps a player expects to be left at any point of a game, the plain time budget estimator sharing the time left between them (see `timing/README.md`); below 1 is refused |
| `--selection ucb1\|puct` | `ucb1` | how a tried node picks the action to follow: `ucb1` tries every legal action once first, `puct` follows Q + c · P · √N / (1 + n) with a prior (see `mcts/README.md`) |
| `--prior uniform\|rater\|value` | `uniform` | the prior PUCT follows: every action alike, the agent's rules' ratings, or its value rules' values of each action's outcomes; `rater` needs `--rules` and `value` needs `--values`; the untrained agent always follows `uniform` |
| `--puct-exploration X` | `1.5` | PUCT's exploration weight c; 0 or more |
| `--prior-temperature X` | `0.1` | the softmax temperature turning ratings or values into a prior, lower following the best more closely; above 0 |
| `--knowledge DIR` | `data/knowledge` | where the knowledge base remembers every game with its models, under `<domain>/` (see `agent/README.md`, `GameMemory`) |
| `--workers N` | half the logical CPUs | worker processes baseline games, reference searches and the positions searched at each budget run in; the report is the same whatever the number, apart from seconds per choice, and every worker holds its own caches, so memory grows with it |
| `--rollouts MODE` | `guided` | with `--rules`: `guided` rollouts follow the rules' ratings; `unguided` rollouts pick uniformly and only the search tree's nodes are rated, which is much cheaper with rules reading lookahead such as `wins()` |
| `--values PATH` | none | value base valuing the positions the evaluated agent's rollouts reach; its path is recorded in the report |
| `--rollout-actions N` | `0` | with `--values`: rollout actions played before valuing a position; 0 values the search's new position itself |
| `--rollout-limit N` | no limit | actions a rollout plays at most, for every agent the evaluation builds (evaluated, unguided, untrained), before every player gets the unfinished payoff; reference searches play to the end |
| `--unfinished-payoff X` | `0.5` | with `--rollout-limit`: each player's payoff for a rollout stopped at the limit |
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
action optimal in <n>)`; with `--rollout-limit N`, it adds `rollout limit N, unfinished payoff X; ` before `every
action optimal`.

It saves the report as
`<report directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, ending with `INFO Saved report <path>` from logger
`openmind.entrypoint.evaluate`.

With `--rules`, the log also has `INFO Evaluating with rules <path>`, and with `--values`,
`INFO Evaluating with values <path>`. With `--rollout-limit`, the logs of `openmind-distill` and
`openmind-distill-values` have `INFO Self-play rollouts stop after <n> actions, every player getting <payoff>`. Every
log starts with
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
| `--rollout-limit N` | no limit | actions a self-play rollout plays at most before every player gets the unfinished payoff; domains whose random games run long, such as chess, need one |
| `--unfinished-payoff X` | `0.5` | with `--rollout-limit`: each player's payoff for a rollout stopped at the limit |
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
| `--seconds X` | `3600.0` | seconds the inference engine's expression search runs (see `inference/README.md`) |
| `--memory X` | half the machine's memory | GB the expression search's process holds at most, measured; its workers each hold an even share; this process's caches are cleared above it |
| `--candidates N` | no limit | candidates the expression search tries at most; a search limited by candidates, unlike one limited by time, gives the same rules on any machine |
| `--training-time-control MINUTES+SECONDS` | no clock | self-play games play on a clock, every agent budgeting its moves; written as chess writes a time control: `3+2` is 3 minutes and 2 seconds a move; `--iterations` then caps each move; a domain without a timeout rule is refused, and so is anything but minutes+seconds or a base of 0 |
| `--expected-steps N` | `30` | with a clock: steps a player expects to be left at any point of a game, the plain time budget estimator sharing the time left between them (see `timing/README.md`); below 1 is refused |
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
| `--unfinished-payoff X` | `0.5` | with `--rollout-limit`: each player's payoff for a rollout stopped at the limit |
| `--deduction-plies N` | `0` | actions ahead a deduction of one position looks at most: self-play agents deduce the positions their rules have no clue about, and pondering deduces the positions missed most; `0` never deduces |
| `--deduction-seconds X` | `10.0` | seconds a deduction of one position runs at most |
| `--highest-payoff X` | `1.0` | the highest payoff a player can get: a move proven to reach it needs no comparison with moves not proven yet |
| `--ponder-positions N` | `0` | training positions the rules missed most, deduced before fitting, their proofs becoming targets and seeds; needs `--deduction-plies` |
| `--ponder-endings N` | `0` | positions of decisive training games deduced at most per round, walking back from each game's end until a position isn't proven, before the positions missed most; their proofs become targets and seeds too; needs `--deduction-plies` |
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
Pondered <n> positions: <p> proven; <s> seeds, <k> kept by the search, <r> in the value rules   (with --ponder-positions)
Saved values <path>
```

`--ponder-positions` or `--ponder-endings` without `--deduction-plies`, `--deduction-seconds` of 0 or less with it, and `--memory` or
`--worker-memory` of 0 or less are rejected, and so is `--target signals`, which needs `openmind-train-values`.

It saves the value base as `<values directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` and writes the log as
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`: it starts with `INFO Running self-play and term evaluations in
<n> worker processes, each holding at most <bytes> bytes; memory diagnoses in <directory>` and `INFO Valuing positions at the <target> target; prices <prices>; searching expressions for <seconds> seconds within
<bytes> bytes, trying any number of|at most <n> candidates`, has the search's generations (see `inference/README.md`) and the generator's fits (see
`rbs/README.md`), and ends with `INFO Saved values <path>` from logger
`openmind.entrypoint.distill_values`.

## `openmind-train-values`

```bash
.venv/bin/openmind-train-values tictactoe --games 20
.venv/bin/openmind-train-values chess --training-time-control 1+0 --rollout-limit 100 --deduction-plies 3 --deduction-seconds 2 --ponder-positions 20 --ponder-endings 20
```

Trains continuously (see "How continuous training works" in `training/README.md`): games between arms one after
another, each game learned from as it ends, a rule search after every decisive game.

| Option | Default | Meaning |
|---|---|---|
| `--games N` | until stopped | games to play, then stop |
| `--iterations N` | `100` | MCTS iterations per move; with a clock, the cap of each move |
| `--training-time-control MINUTES+SECONDS` | no clock | games play on a clock, every agent budgeting its moves; written as chess writes a time control: `3+2` is 3 minutes and 2 seconds a move; a domain without a timeout rule is refused, and so is anything but minutes+seconds or a base of 0 |
| `--expected-steps N` | `30` | with a clock: steps a player expects to be left at any point of a game, the plain time budget estimator sharing the time left between them (see `timing/README.md`); below 1 is refused |
| `--selection ucb1\|puct` | `ucb1` | how a tried node picks the action to follow: `ucb1` tries every legal action once first, `puct` follows Q + c · P · √N / (1 + n) with a prior (see `mcts/README.md`) |
| `--prior uniform\|rater\|value` | `uniform` | the prior PUCT follows: every action alike, or each arm's own value rules' values of each action's outcomes; `rater` is refused, training having no rules that rate actions |
| `--puct-exploration X` | `1.5` | PUCT's exploration weight c; 0 or more |
| `--prior-temperature X` | `0.1` | the softmax temperature turning values into a prior, lower following the best more closely; above 0 |
| `--knowledge DIR` | `data/knowledge` | where the knowledge base remembers every game with its models, every proof and every seed, under `<domain>/` |
| `--seed S` | `1` | random seed |
| `--learning-rate X` | `0.01` | how far each weight moves toward what a finished game showed; below 0 is refused |
| `--seconds X` | `600` | seconds the rule search after each decisive game runs at most |
| `--memory X`, `--candidates N`, `--prices LIST`, `--max-steps N`, `--tolerance X` | as `openmind-distill-values` | how each rule search searches and fits; the middle price also prices each game's weight step |
| `--rollout-actions N` | `10` | rollout actions played before a position is valued with value rules |
| `--rollout-limit N` | no limit | actions a rollout plays at most before every player gets the unfinished payoff |
| `--unfinished-payoff X` | `0.5` | with `--rollout-limit`: each player's payoff for a rollout stopped at the limit |
| `--deduction-plies N`, `--deduction-seconds X`, `--highest-payoff X` | as `openmind-distill-values` | the deduction every agent falls back on when its rules have no clue, which pondering needs |
| `--ponder-positions N` | `0` | positions of each game the reference rules missed most, deduced in the game's study; needs `--deduction-plies` |
| `--ponder-endings N` | `0` | positions of each decisive game deduced at most, walking back from its end until one isn't proven; needs `--deduction-plies` |
| `--workers N` | half the logical CPUs | worker processes games are played and studied in, and the rule search evaluates terms in |
| `--worker-memory X` | `--memory` shared between the workers | as `openmind-distill-values`; a game dropped over it isn't learned from |
| `--arms N` | `8` | signals followed at most, those with the best records, besides winning and the aggregations |
| `--signal-horizon N` | `0` | plies later a position's signals are read for its targets |
| `--arm-exploration X` | `1.414...` (√2) | UCB1's exploration weight when a worker taking a game chooses which arms play it; below 0 is rejected |
| `--goal-limit N` | `2` | how many moves ahead the deduced goal distance looks for a win when signals are deduced; below 1 is rejected |
| `--signal-library PATH` | the deduced signals | the signal library to start from, such as an earlier run's; a library for another domain is rejected |
| `--signals-directory DIR` | `data/signals` | where the signal library is saved after every game, as `<directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.json` named after the training's start |
| `--log-level LEVEL` | `INFO` | lowest level saved in the log |
| `--log-directory DIR` | `data/log/train-values` | where logs are saved |

After every game it saves the signal library, its records, rules and arms, so a training stopped at any moment keeps
every finished game's learning; `--signal-library` carries on from it, and the games are numbered after those the
knowledge base already remembers. At the end it prints `Saved signal library <path>`. The log,
`<log directory>/<domain>/<YYYY-MM-DD_HH-MM-SS>.log`, starts with `INFO Training in <n> worker processes, each holding
at most <bytes> bytes; memory diagnoses in <directory>` and `INFO Following signals: ...`, then has the training's lines
(see `training/README.md`), and ends with `INFO Saved signal library <path>`, from logger `openmind.entrypoint.train_values`.

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
  `test/end_to_end/distill_values_tictactoe_tests.py`, `test/end_to_end/train_values_tictactoe_tests.py`,
  `test/end_to_end/solve_sudoku_tests.py`.
