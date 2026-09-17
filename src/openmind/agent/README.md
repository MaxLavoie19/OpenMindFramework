# agent

## Purpose

The agent and the domains it works on. The agent chooses actions by searching with MCTS. Tic-tac-toe and sudoku are
domains within the agent, not domains in the code: they exist only as this domain's factory recipes and their
constants.

## Content

| File | What it is |
|---|---|
| `model/domain.py` | `Domain(name, initial_state, problem, transitions, players, observation=None, ending=None, record=None, timeout=None, picture=None)`: a domain within the agent; `observation` says what each player sees of a state (see `observation/README.md`), `None` when every player sees everything; `ending`, a rule reading a finished game's last state, says why it ended, and `record`, a rule reading the initial state and the parameter `actions`, gives the game's record as one line, both for the logs and, when they are source, seeing the transitions' definitions; either may be the project's own function (`EndingRule`, `RecordRule`); `timeout`, an effects rule reading `flagged`, the name of the player whose clock ran out, gives the state with the payoffs that sets, and without it the domain can't be played on a clock; `picture`, a rule reading a state and `last`, the action that led to it (None at the start), gives an SVG image of the position as text, for the dashboard's games, positions being shown as text without it |
| `builder/domain_builder.py` | `DomainBuilder`: collects a domain's parts, `with_observation`, `with_ending`, `with_record`, `with_timeout` and `with_picture` optional; rejects missing parts, and rejects a rule written as a function no worker process could find, such as a lambda or a function defined inside another (see `rule/README.md`) |
| `constant/game_record_constant.py` | `ACTIONS` (`"actions"`), the parameter a record rule reads a game's actions from |
| `service/timekeeper.py` | `Timekeeper(rule_caller, time_source=None)`, what a referee needs to keep the players' clocks, on wall time unless given another source: `clocks(domain, control)`, each player's starting clock, raising `ValueError` for a domain without a timeout rule; `timed(choose)`, what a choice gave and the seconds it took; `flag(domain, state, player)`, the state after the player's clock ran out, by the domain's timeout rule |
| `service/game_recorder.py` | `GameRecorder(rule_caller)`: `ending(domain, state)` and `record(domain, actions)`, what the domain's rules say about a finished game; `None` for a domain without the rule, and for a rule that raises, with a warning |
| `factory/domain_factory.py` | `create_domain(name)`: creates a domain from its name (`"tictactoe"`, a variant such as `"tictactoe/fourinarow"`, `"sudoku"`, or a domain an installed project registers); an unknown name or variant raises `ValueError` listing the known ones |
| `model/domain_recipe.py` | `DomainRecipe`: what an installed project registers, a function giving its domain from the whole domain name |
| `service/agent.py` | `Agent`: searches a domain's state with MCTS, guided by a rater and valuing positions with a valuer when built with them, falling back on deduction when built with a budget, and in a domain with an observation searching semi-determinized when built with a theory of mind (see `mcts/README.md`); where players act at once, it searches for the player given to `search(domain, state, player)` or `choose(domain, state, player)`, the other players to act playing the strategies its theory of mind predicts, if any, and samples its action from its average strategy; `search` gives the whole result, `choose` the action. Given a clock and the steps its player has played (`search(domain, state, player, clock, steps_played)`), it asks its time budget estimator how long the step may take and searches for that long, any iterations it was built with kept as a cap; without a clock it searches its iterations. A clock without an estimator, or no clock for an agent built without iterations, raise `ValueError`. The deduction fallback keeps its own seconds, not bounded by the step's budget |
| `service/completion_theory.py` | `CompletionTheory(state_observer, rule_compiler, rule_runner, label=None)`: a theory of mind that knows nothing about the other players and believes the domain's completions; the states that could be true are grouped into hypotheses by the values of every hidden variable, or by a label rule reading such a state, `player` and the observation's definitions, each hypothesis weighted by its states' summed probabilities and its states renormalized |
| `service/deduction_fallback.py` | `DeductionFallback.result(domain, state, valuation)`: in a domain every player sees whole, when the agent has no valuer, or its valuer can't value every legal action's outcomes or values them all the same for the player to act, deduces the position (see `inference/README.md`); a proven best action comes back as a search result holding that action alone, with 1 visit and its proven payoff, and no samples; otherwise `None`, and the agent searches |
| `model/policy.py` | `Policy`: anything with `choose(domain, state, player=None, clock=None, steps_played=0) -> Action`, the player given where players act at once, and on a clock the player's clock and the steps it has played so far; `Agent` and `RandomPolicy` are policies |
| `model/describable.py` | `Describable`: a part of an agent that can say what it is, `describe() -> str`, JSON enough to build it again; `RuleValuer` gives its value base, `PlainTimeBudgetEstimator` its rule and steps expected |
| `model/model_description.py` | `ModelDescription(name, text)`: a model as it played, `text` everything needed to build it again word for word; `id`, the first 16 hex digits of the text's SHA-256, is the same for the same model and changes with any setting or rule |
| `model/game_summary.py` | `GameSummary(domain, kind, round, number, seeds, players, models, payoffs, plies, ending=None, record=None, time_control=None, seconds=(), budgets=(), clocks=(), flagged=None)`: a finished game as it is remembered, whatever played it, the model each player played in the players' order; `label` names it, `round 1 arms game 12` or `match game 3` |
| `mapper/game_summary_json_mapper.py` | `GameSummaryJsonMapper`: a game summary as JSON and back, each model kept by name and id; reading one back takes the models by id and raises `ValueError` for one not given |
| `service/game_memory.py` | `GameMemory(knowledge_base)`: `remember(summary)` keeps a finished game in the knowledge base as it ends — each model once, its text word for word under its id and name with the keyword `model`; the game, its summary as JSON with the keyword `game` and its kind; and each player's outcome, `win`, `draw` or `loss` as the keyword, under that player's model's id and name, reading like `deduced, losing color doubled drew as white in round 1 arms game 12` — every record `played`, with the game and its round. The highest payoff alone wins, a highest payoff shared draws, anything lower loses. Outcomes are facts, not evidence for a claim: a draw is a draw. `scores(names)` counts each name's games, wins, draws and losses (a model playing both sides counts both); `models()` gives every model remembered |
| `model/policy_factory.py` | `PolicyFactory`: gives the policy that plays a game from the game's seed; to run in worker processes, it must pickle |
| `service/random_policy.py` | `RandomPolicy`: chooses uniformly among the legal actions, the given player's where players act at once; a clock changes nothing; a baseline opponent |
| `builder/agent_builder.py` | `AgentBuilder`: sets iterations, exploration, seed, guidance (`with_guidance(rater)`), whether guided rollouts follow the ratings (`with_guided_rollouts(guided)`), the valuer valuing the positions rollouts reach (`with_valuation(valuer)`), the rollout actions played before valuing (`with_rollout_actions(actions)`, 0 by default), the rollout limit (`with_rollout_limit(limit, unfinished_payoff)`) the deduction the agent falls back on when its rules have no clue (`with_deduction(budget)`, none by default) and the theory of mind a semi-determinized search asks for hypotheses (`with_theory_of_mind(theory=None)`, `CompletionTheory` when none is given; without the call the agent searches plain information set MCTS) how it budgets a step's time on a clock (`with_time_budget_estimator(estimator)`, with which iterations become an optional cap), and how its search selects (`with_selection(selection, puct_exploration=1.5)`, `ucb1` by default, and `with_prior(prior)` for PUCT), and wires the services the agent searches with; `describe(name)` gives the agent it builds as a `ModelDescription`, every setting but the seed as JSON, each model as it describes itself and one that can't as its class marked `not rebuildable`; rejects a missing exploration, neither iterations nor an estimator, fewer than 1 iteration, negative rollout actions, a negative rollout limit or one without an unfinished payoff, and a deduction budget without plies or seconds |
| `factory/agent_factory.py` | `create_agent(iterations=1000, seed=None, rollout_limit=None, unfinished_payoff=None)`: an agent searching with the exploration weight √2 |
| `constant/agent_constant.py` | Default iterations (1000), exploration weight (√2), the guidance's prior weight (1.0), rollout temperature (0.2) and guided rollouts (true), the default unfinished payoff (0.5), and the entry point group installed domains register under (`openmind.domains`) |
| `model/tictactoe_variant.py` | `TicTacToeVariant(name, width, height, line, gravity)`: how a variant differs from standard tic-tac-toe |
| `constant/tictactoe_constant.py` | Domain name and the variant separator, players, empty and unset values, payoff values, variable and action names, the four line directions, and the variants (`STANDARD`, `VARIANTS`) |
| `factory/tictactoe_factory.py` | `create_tictactoe_domain(variant=STANDARD)`, assembled from `create_tictactoe_initial_state(variant)`, `create_tictactoe_problem(variant)`, `create_tictactoe_transitions(variant)` `create_tictactoe_players()` and `create_tictactoe_timeout()`, on a clock the player whose time ran out losing, with `create_tictactoe_definitions(variant)` giving the script every rule of the variant sees; a variant without room for its line raises `ValueError` |
| `constant/sudoku_constant.py` | Domain name and the separator of puzzle names, box and grid size, digits, the puzzle, empty and clue marks, empty and unset values, the collection file suffix and Project Euler's format marks, payoff values, variable and action names, and the parameter name template (`cell_{row}_{col}`) |
| `factory/sudoku_factory.py` | `create_sudoku_domain(name="sudoku", grid=PUZZLE)`, assembled from `create_sudoku_initial_state(grid)`, `create_sudoku_problem(grid)`, `create_sudoku_transitions(grid)` and `create_sudoku_players()` |
| `model/sudoku_puzzle.py` | `SudokuPuzzle(collection, number, grid)`: a published puzzle, numbered from 1 in its collection, its grid 81 characters row by row with `.` for an empty cell |
| `mapper/sudoku_collection_mapper.py` | `SudokuCollectionMapper`: reads a collection's text into puzzles, from one 81-character line per puzzle (Norvig) or a `Grid NN` line and 9 rows (Project Euler), with `.` or `0` for an empty cell; anything else raises `ValueError` |
| `repository/sudoku_puzzle_repository.py` | `SudokuPuzzleRepository`: lists the `<collection>.txt` files of a directory and loads a collection's puzzles |
| `model/prisoners_dilemma_variant.py` | `PrisonersDilemmaVariant(name, rounds, ending_chance, simultaneous=False)`: how many rounds are played (`None` when no last round is known), the chance the game ends after each round, and whether both players choose at once |
| `constant/rock_paper_scissors_constant.py` | Domain name, players, the three shapes and what each beats, the payoffs of a win (1), draw (0.5) and loss (0), variable, action and parameter names |
| `factory/rock_paper_scissors_factory.py` | `create_rock_paper_scissors_domain()`, assembled from `create_rock_paper_scissors_initial_state()`, `create_rock_paper_scissors_problem()`, `create_rock_paper_scissors_transitions()` and `create_rock_paper_scissors_players()`, with `create_rock_paper_scissors_definitions()` giving the script its rules see |
| `constant/prisoners_dilemma_constant.py` | Domain name and the variant separator, players, the two choices, Axelrod's points (`REWARD`, `PUNISHMENT`, `TEMPTATION`, `SUCKER`, `POINTS`), variable, action and parameter names, and the variants (`STANDARD`, `VARIANTS`) |
| `factory/prisoners_dilemma_factory.py` | `create_prisoners_dilemma_domain(variant=STANDARD)`, assembled from `create_prisoners_dilemma_initial_state()`, `create_prisoners_dilemma_problem()`, `create_prisoners_dilemma_transitions(variant)` and `create_prisoners_dilemma_players()`, with `create_prisoners_dilemma_definitions(variant)` giving the script its rules see; a variant with fewer than 1 round, an ending chance outside 0 to 1, or neither a last round nor an ending chance raises `ValueError` |

## Domains from installed projects

OpenMind ships without the libraries a problem needs. A problem is programmed in its own project, which installs
OpenMind, builds its `Domain` with OpenMind's classes, its rules free to import any library, and registers a recipe in
its `pyproject.toml`:

```toml
[project.entry-points."openmind.domains"]
chess = "openmind_chess.game.factory.chess_factory:create_chess_domain"
```

`create_domain(name)` looks in its own domains first, then in the installed recipes for the part of the name before
`/`, and calls that recipe with the whole name (`"chess"`, or a variant such as `"chess/960"`). Every command taking a
domain name, `openmind-play` and `openmind-evaluate` among them, then runs the project's domain.

## Tic-tac-toe and its variants

A variant, `TicTacToeVariant(name, width, height, line, gravity)`, describes how a game differs from standard
tic-tac-toe: a grid of `width` columns by `height` rows, `line` marks in a row to win, and, with `gravity`, marks that
fall to the lowest empty cell of the column they are dropped in. The variants are data in
`tictactoe_constant.VARIANTS`, each written as the standard game plus what it changes; the same recipes build them all.

| Variant | Domain name | Width × height | Line | Gravity |
|---|---|---|---|---|
| `standard` | `tictactoe` | 3 × 3 | 3 | no |
| `fourinarow` | `tictactoe/fourinarow` | 7 × 6 | 4 | yes |
| `gomoku` | `tictactoe/gomoku` | 15 × 15 | 5; a longer line also wins (freestyle) | no |

A variant needs a width and a height of at least 1 and a line from 1 to its longer side; otherwise the recipes raise
`ValueError`. A variant that changes a rule rather than a size, such as misère, would add a field that the recipes
read.

### Players

`Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)"))`: `turn` names the player to act, and `payoff(X)` and
`payoff(O)` hold the players' payoffs.

### State variables

| Variable | Values | Initial |
|---|---|---|
| `cell(row,col)`, row in 1..height from the top, col in 1..width | `"X"`, `"O"`, or `None` when empty | `None` |
| `turn` | `"X"` or `"O"` | `"X"` |
| `payoff(X)`, `payoff(O)` | 1 for a win, 0 for a loss, 0.5 each for a draw; `None` until the game ends | `None` |

### Constraints (CSP)

Every rule of a variant sees the names of `create_tictactoe_definitions(variant)`, a script run once: `WIDTH`,
`HEIGHT`, `LINE`, `PLAYERS`, `WIN`, `DRAW`, `LOSS`, `other(player)`, and `LINES_THROUGH[row, col]`, every line of
`LINE` cells through a cell that fits the grid.

Without gravity, the action `place(row, col)`, with row in 1..height and col in 1..width, is legal when these rules
hold, checked in this order:

```python
payoff['X'] is None
payoff['O'] is None
cell[row, col] is None
```

With gravity, the action `drop(col)`, with col in 1..width, has the same payoff rules and `cell[1, col] is None`, the
column's top cell.

### Transitions (predictor)

The action has one branch with probability 1, whose effects are this script:

```python
cell[row, col] = turn
if any(all(cell[r, c] == turn for r, c in line) for line in LINES_THROUGH[row, col]):
    payoff[turn] = WIN
    payoff[other(turn)] = LOSS
elif all(mark is not None for mark in cell.values()):
    payoff['X'] = DRAW
    payoff['O'] = DRAW
turn = other(turn)
```

With gravity it first finds the landing row, `row = max(r for r in range(1, HEIGHT + 1) if cell[r, col] is None)`, and
the draw check reads only the top row: `all(cell[1, c] is not None for c in range(1, WIDTH + 1))`.

Only the lines through the landing cell are checked, so a move in 4 in a row checks at most 13 lines rather than all
69.

## Sudoku

Sudoku is solved by the CSP alone, without MCTS or RBS: `openmind-solve sudoku`. The domain `sudoku` holds
`sudoku_constant.PUZZLE`, row by row, with `.` for an empty cell. `create_sudoku_domain(name, grid)` makes a domain of
any other grid; a grid that isn't 81 characters, each `.` or a digit from 1 to 9, raises `ValueError`.

Published collections are read from `data/sudoku/`, one `<collection>.txt` file each (the root README has the
downloads). `openmind-solve` names their puzzles `sudoku/<collection>/<number>`:

```python
from pathlib import Path

from openmind.agent.factory.sudoku_factory import create_sudoku_domain
from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.repository.sudoku_puzzle_repository import SudokuPuzzleRepository

repository = SudokuPuzzleRepository(SudokuCollectionMapper())
repository.collections(Path("data/sudoku"))   # ('euler', 'hardest', 'top95')
puzzle = repository.load(Path("data/sudoku"), "top95")[6]
domain = create_sudoku_domain(f"sudoku/{puzzle.collection}/{puzzle.number}", puzzle.grid)
```

### Players

`Players(("solver",), "turn", ("payoff",))`.

### State variables

| Variable | Values | Initial |
|---|---|---|
| `cell(row,col)`, row and col in 1..9 | a digit 1..9, or `None` when empty | the grid's clue, or `None` |
| `turn` | `"solver"` | `"solver"` |
| `payoff` | 1.0 once the grid is filled; `None` until then | `None` |

### Constraints (CSP)

The action `fill` has one parameter per empty cell, named `cell_<row>_<col>` (`cell_1_3`), with domain 1..9. It is
legal when `payoff is None` holds and, for each of the 27 rows, columns and boxes, an `all_different` rule over its
cells, the parameters of its empty cells and the clues read from the state:
`all_different(cell[1, 1], cell[1, 2], cell_1_3, cell_1_4, cell[1, 5], cell_1_6, cell_1_7, cell_1_8, cell_1_9)`.

### Transitions (predictor)

`fill` has one branch with probability 1, a script writing every parameter into its cell (`cell[1, 3] = cell_1_3`, one
line per empty cell), then `payoff = 1.0`, the share of cells filled.

## Repeated prisoner's dilemma

The first non-zero-sum game: each player's payoff is their own total, so both can gain, or both lose. The two players
choose at the same time, which the `standard` and `uncertain` variants play as A choosing, then B, with each player's
choice hidden from the other by the domain's observation, and the `simultaneous` variant plays with both players acting
at once.

| Variant | Domain name | Rounds | Ending chance after each round | Choices |
|---|---|---|---|---|
| `standard` | `prisonersdilemma` | 10 | 0 | A then B, hidden |
| `uncertain` | `prisonersdilemma/uncertain` | no known last round | 0.1 | A then B, hidden |
| `simultaneous` | `prisonersdilemma/simultaneous` | 10 | 0 | at once |

In the `simultaneous` variant, the state has `turn(A)` and `turn(B)` instead of `turn`, true while the game goes on;
`choose` is legal for a player to act when `turn[player]` holds and `chosen[player] is None`; its effects are only
`chosen[player] = choice`, and the transition model's resolution plays the round with the script below, from `points =`
on, with `each` for `player`, clearing both turns when the game ends. It has no observation: a choice is kept only until
the resolution, which runs before anyone chooses again.

With a known last round, perfect play defects in every round, whatever the ending chance. Without one, the game can't
be searched exactly, since its states never stop.

### Players

`Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)"))`.

### State variables

| Variable | Values | Initial |
|---|---|---|
| `chosen(A)`, `chosen(B)` | this round's choice, `"cooperate"` or `"defect"`; `None` until made, and again once the round is played | `None` |
| `played(round,player)` | the choice played in a round; `None` for the round being chosen; the variables of a round are added when it starts | `played(1,A)` and `played(1,B)`, `None` |
| `round` | the round being chosen, from 1 | 1 |
| `score(A)`, `score(B)` | points so far | 0 |
| `turn` | `"A"` or `"B"` | `"A"` |
| `payoff(A)`, `payoff(B)` | the player's score when the game ends; `None` until then | `None` |

### Constraints (CSP)

Every rule sees `create_prisoners_dilemma_definitions(variant)`: `PLAYERS`, `POINTS[choice of A, choice of B]`, the
points of both players for a round, `ROUNDS`, `None` without a known last round, and `other(player)`. The action
`choose(choice)`, with choice `"cooperate"` or `"defect"`, is legal when these rules hold:

```python
payoff['A'] is None
payoff['B'] is None
chosen[turn] is None
```

### Transitions (predictor)

| A | B | Points of A | Points of B |
|---|---|---|---|
| cooperate | cooperate | 3 | 3 |
| cooperate | defect | 0 | 5 |
| defect | cooperate | 5 | 0 |
| defect | defect | 1 | 1 |

`choose` has one branch without an ending chance, `ENDING = False`, or two: the game goes on with 1 − the chance and
ends with the chance, `ENDING = True`. Each branch runs this script after setting `ENDING`:

```python
chosen[turn] = choice
if all(chosen[player] is not None for player in PLAYERS):
    points = POINTS[chosen[PLAYERS[0]], chosen[PLAYERS[1]]]
    for player, gained in zip(PLAYERS, points):
        played[round, player] = chosen[player]
        score[player] = score[player] + gained
        chosen[player] = None
    if round == ROUNDS or ENDING:
        for player in PLAYERS:
            payoff[player] = score[player]
    else:
        round = round + 1
        for player in PLAYERS:
            played[round, player] = None
turn = other(turn)
```

The next round's `played` variables are new variables the state gains (see `rule/README.md`). A choice that doesn't
finish a round gives the same state on both branches.

### Observation

`create_prisoners_dilemma_observation(variant)` hides the other player's choice, with the variant's definitions:

```python
(chosen_name(other(player)),)                                   # hidden
[({chosen_name(other(player)): choice}, 1 / len(CHOICES)) for choice in CHOICES] \
    if player == turn == PLAYERS[1] else [({chosen_name(other(player)): None}, 1.0)]   # completions
```

On B's turn, B sees `chosen(A) = '<hidden>'`, which could be either choice at even chances; A, choosing first, has
nothing to guess, since B hasn't chosen. The played rounds, scores and payoffs stay visible to both. An agent searches
from its view (see `mcts/README.md`); exact search refuses the domain.

## Rock paper scissors

The first domain where players act at once: A and B throw a shape together, rock beating scissors, paper beating rock
and scissors beating paper. A win pays 1, a draw 0.5 each, a loss 0. Its only equilibrium is throwing each shape a third
of the time, which the search's regret matching heads toward; an opponent predicted to favour a shape gets the shape
beating it.

### Players

`Players(("A", "B"), "turn", ("payoff(A)", "payoff(B)"))`, with `turn(A)` and `turn(B)` flagging the players to act.

### State variables

| Variable | Values | Initial |
|---|---|---|
| `hand(A)`, `hand(B)` | the shape thrown, `"rock"`, `"paper"` or `"scissors"`; `None` until thrown | `None` |
| `turn(A)`, `turn(B)` | whether the player is to act | `True` |
| `payoff(A)`, `payoff(B)` | 1, 0.5 or 0 once both have thrown; `None` until then | `None` |

### Constraints (CSP)

Solved for each player to act, read as `player`: `throw(shape)`, with shape `"rock"`, `"paper"` or `"scissors"`, is
legal when `turn[player]` holds and `hand[player] is None`.

### Transitions (predictor)

`throw` sets `hand[player] = shape`. The resolution, run once both have thrown, compares `hand['A']` and `hand['B']`
with `BEATS`, sets both payoffs, and sets both turns to `False`, ending the game.

## Usage

```python
from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.factory.domain_factory import create_domain

domain = create_domain("tictactoe")
action = create_agent(iterations=500, seed=1).choose(domain, domain.initial_state)

fourinarow = create_domain("tictactoe/fourinarow")   # or create_tictactoe_domain(VARIANTS["fourinarow"])
```

Using the solver and predictor directly:

```python
from openmind.agent.factory.domain_factory import create_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.predictor.factory.predictor_factory import create_predictor

domain = create_domain("tictactoe")
actions = create_solver().solve(domain.problem, domain.initial_state)
# 9 actions; actions[0] is Action(name='place', parameters=(('col', 1), ('row', 1)))
distribution = create_predictor().predict(domain.transitions, domain.initial_state, actions[0])
# OutcomeDistribution(outcomes=((State(variables=(('cell(1,1)', 'X'), ('cell(1,2)', None), …, ('turn', 'O'))), 1.0),))
```

## Notes

- Tests: `builder/agent_builder_tests.py`, `builder/domain_builder_tests.py`, `factory/agent_factory_tests.py`,
  `factory/domain_factory_tests.py`, `factory/prisoners_dilemma_factory_tests.py`, `factory/rock_paper_scissors_factory_tests.py`, `factory/sudoku_factory_tests.py`,
  `factory/tictactoe_factory_tests.py`,
  `mapper/sudoku_collection_mapper_tests.py`, `repository/sudoku_puzzle_repository_tests.py`,
  `service/agent_tests.py`, `service/random_policy_tests.py`; integration:
  `test/integration/tictactoe_actions_tests.py`, `test/integration/tictactoe_search_tests.py`,
  `test/integration/tictactoe_transitions_tests.py`, `test/integration/tictactoe_variants_games_tests.py`,
  `test/integration/tictactoe_fourinarow_transitions_tests.py`, `test/integration/tictactoe_fourinarow_search_tests.py`,
  `test/integration/sudoku_solve_tests.py`, `test/integration/sudoku_collections_solve_tests.py`,
  `test/integration/prisoners_dilemma_transitions_tests.py`, `test/integration/prisoners_dilemma_search_tests.py`;
  end-to-end:
  `test/end_to_end/play_tictactoe_tests.py`, `test/end_to_end/play_fourinarow_tests.py`,
  `test/end_to_end/solve_sudoku_tests.py`.
