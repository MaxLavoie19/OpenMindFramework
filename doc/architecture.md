# Architecture

## Vocabulary

- **State**: a set of named variables with values, such as `cell(2,3) = "X"` or `turn = "O"`.
- **Action**: the thing performed (dance, sing, move, …) with its parameters, such as `place(row=2, col=3)`.
- **Outcome**: the new state after an action.
- **Outcome probability distribution**: each possible outcome of an action with its probability, given by the
  predictor.
- **Payoff**: what each player gets when the game ends; a predicted outcome of a transition to a win or a draw.
- **Domain (within the agent)**: a problem the agent works on, such as tic-tac-toe. It is not a domain in the code.
- **Variant**: a game that differs from another in some of its sizes or rules, such as 4 in a row from tic-tac-toe.
  A variant is data read by the game's recipes, and its domain is named `<game>/<variant>`, such as
  `tictactoe/fourinarow`.

## Rules of a domain

| Rule | Lives in |
|---|---|
| Constraints on actions and their parameters | the domain's constraint satisfaction problem (CSP) |
| Transitions, end of game and payoffs | the predictor |
| Strategy, weighted by expected value | the strategy rule-based system (RBS) |
| Players and initial state | the domain's factory |

Rules are Python: every constraint, effect and RBS condition is the source of a Python expression or script, compiled
once and run against states (see `src/openmind/rule/README.md`). Any Python is allowed, imports and libraries included,
because OpenMind is a general-purpose framework that people use to write the rules of their own problems; nothing in
the framework is specific to one game. A domain's definitions script runs once and gives all of its rules shared
names. Factories write rules for now; later, a decoder will turn unstructured data into rules.

## Code domains

| Domain | Owns | Status |
|---|---|---|
| `world` | `Value`, `State`, `Action`, and their readable text | iterations 1 and 8 |
| `rule` | Python rules: compiling them, running them against states, and a state as the names a rule reads | iteration 9 |
| `csp` | Constraint satisfaction: action definitions, domains fixed or computed from the state, constraints, and a solver with propagation and backtracking | iterations 1, 7, 9 and 12 |
| `agent` | The agent and its domains: tic-tac-toe with its variants, sudoku, the repeated prisoner's dilemma with its variants, and the domains installed projects register | iterations 4, 7, 8 and 12 |
| `entrypoint` | Ways to run the framework: `openmind-play`, `openmind-evaluate`, `openmind-distill`, `openmind-select`, `openmind-distill-values`, `openmind-solve` | iterations 3–7, 10 and 11 |
| `predictor` | Transitions, outcome probability distributions | iterations 2 and 9 |
| `observation` | What each player sees of a state, and the states that could be true given what they see | hidden information |
| `mcts` | Monte-Carlo Tree Search, optionally guided by a model behind `ActionRater`, valuing positions with a model behind `PositionValuer`, and stopping rollouts at a limit | iterations 4–6, 11 and 12 |
| `evaluation` | Measures how well an agent plays: baselines, agreement with perfect play or a reference search, paired tests of guidance, rules and values alone | iterations 5, 8, 10 and 11 |
| `rbs` | Rules generated from search for any domain, as hypotheses validated on held-out games, that rate actions and explain their ratings; value rules, weighted terms fitted sparsely, that value positions | iterations 6, 9, 10 and 11 |
| `inference` | The inference engine: views of positions that look ahead with a domain's own actions, a search growing expressions of them (patterns of any size, thresholds, combinations, look-aheads) within a time and memory budget, and deduction on one position with induction of candidate expressions from what it proved | iteration 14 |
| `training` | Self-play, distillation of models from search, selection of the rules that play no worse than all of them, distillation of value rules, and a loop training value rules round after round | iterations 6, 10, 11 and 13 |
| `parallel` | Running independent games and searches in worker processes, results in order, logs forwarded; each process's caches cleared by memory, and workers capped, ended with a diagnosis and replaced when they stay over | iteration 10, 2026-09-15 |
| `dashboard` | A page following a value training while it runs: the round's progress, every round, the latest rules, the machine's memory and earlyoom's kills | 2026-09-15 |
| `timing` | Time as agents and referees see it: time sources, deadlines, time controls and each player's clock; seconds on a timeline, not counts of moves | 2026-09-16 |
| `doxastic` | What the agent remembers, word for word with where it came from, looked up by subject, name, keyword, claim, teller or source; and where each claim stands, the evidence for it and the evidence against it kept apart | 2026-09-15 |
| `testing` | Test support shared with problem projects: a pytest plugin saving each test's logs | iteration 12 |
| `optimizer` | Strategic discrete actions from continuous action spaces | later |

## Conventions

- Code lives in `src/openmind/<domain>/<class type>/`. Class types: `model` (datatypes), `factory` (functions that
  build an object with a builder and a recipe), `builder`, `service` (operations on data), `repository` (storing and
  retrieving data), `mapper` (format conversions), `constant` and `plugin` (pytest plugins). Entrypoints live in
  `src/openmind/entrypoint/`.
- OpenMind ships without the libraries a problem needs. A problem lives in its own project, which installs OpenMind,
  imports any library its rules need, and registers its domains under the `openmind.domains` entry points (see
  `src/openmind/agent/README.md`); chess lives in the OpenMindChess project, with python-chess.
- Each domain folder has a `README.md` covering its purpose, content, usage and logs.
- One class per module, named after it; a name that clashes with a Python keyword gets a trailing underscore
  (`not_.py`).
- Models are frozen, slotted dataclasses, except the search tree's nodes, which change while searching. Logic lives in
  services; I/O only in entrypoints.
- Services log through `logging.getLogger(__name__)`: what was decided and why, with the key values. Actions appear
  in logs as readable text (`ActionTextMapper`) and rules as their source. Per-call details (solver
  candidates, predictor effects, search iterations) log at DEBUG; decisions (choices, search results) at INFO.
- Unit tests sit beside their target as `<module>_tests.py`; integration and end-to-end tests live in `test/`.
- Work that can run in worker processes goes through `parallel`'s `TaskRunner`. Services that keep caches leave them
  behind when pickled, so any service can travel to a worker; work in workers draws its seeds up front, so results
  don't depend on the number of workers.
- `data/` holds all data (databases, trained models, logs, …); its layout is decided as we go. Evaluation reports go in
  `data/evaluation/<domain>/`, selection reports in `data/selection/<domain>/`, rule bases in `data/rbs/<domain>/`, value
  bases in `data/values/<domain>/`, training reports in `data/training/<domain>/` and published sudoku collections in
  `data/sudoku/`,
  all ignored by git. Tests save their logs, through the `openmind.testing.plugin.log_saving` plugin,
  in `data/log/<test file>/<test name>.log`, which git ignores; a test marked `@pytest.mark.log_level("INFO")` saves
  only INFO and above.
