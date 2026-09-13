# agent

## Purpose

The agent and the domains it works on. The agent chooses actions by searching with MCTS. Tic-tac-toe and sudoku are
domains within the agent, not domains in the code: they exist only as this domain's factory recipes and their
constants.

## Content

| File | What it is |
|---|---|
| `model/domain.py` | `Domain(name, initial_state, problem, transitions, players)`: a domain within the agent |
| `builder/domain_builder.py` | `DomainBuilder`: collects a domain's parts; rejects missing parts |
| `factory/domain_factory.py` | `create_domain(name)`: creates a domain from its name (`"tictactoe"`, a variant such as `"tictactoe/fourinarow"`, or `"sudoku"`); an unknown name or variant raises `ValueError` listing the known ones |
| `service/agent.py` | `Agent`: searches a domain's state with MCTS, guided by a rater when built with one; `search` gives the whole result, `choose` the action |
| `model/policy.py` | `Policy`: anything with `choose(domain, state) -> Action`; `Agent` and `RandomPolicy` are policies |
| `service/random_policy.py` | `RandomPolicy`: chooses uniformly among the legal actions; a baseline opponent |
| `builder/agent_builder.py` | `AgentBuilder`: sets iterations, exploration, seed and guidance (`with_guidance(rater)`), and wires the services the agent searches with; rejects missing settings and fewer than 1 iteration |
| `factory/agent_factory.py` | `create_agent(iterations=1000, seed=None)`: an agent searching with the exploration weight √2 |
| `constant/agent_constant.py` | Default iterations (1000), exploration weight (√2), and the guidance's prior weight (1.0) and rollout temperature (0.2) |
| `model/tictactoe_variant.py` | `TicTacToeVariant(name, width, height, line, gravity)`: how a variant differs from standard tic-tac-toe |
| `constant/tictactoe_constant.py` | Domain name and the variant separator, players, empty and unset values, payoff values, variable and action names, the four line directions, and the variants (`STANDARD`, `VARIANTS`) |
| `factory/tictactoe_factory.py` | `create_tictactoe_domain(variant=STANDARD)`, assembled from `create_tictactoe_initial_state(variant)`, `create_tictactoe_problem(variant)`, `create_tictactoe_transitions(variant)` and `create_tictactoe_players()`; a variant without room for its line raises `ValueError` |
| `constant/sudoku_constant.py` | Domain name and the separator of puzzle names, box and grid size, digits, the puzzle, empty and clue marks, empty and unset values, the collection file suffix and Project Euler's format marks, payoff values, variable and action names |
| `factory/sudoku_factory.py` | `create_sudoku_domain(name="sudoku", grid=PUZZLE)`, assembled from `create_sudoku_initial_state(grid)`, `create_sudoku_problem(grid)`, `create_sudoku_transitions(grid)` and `create_sudoku_players()` |
| `model/sudoku_puzzle.py` | `SudokuPuzzle(collection, number, grid)`: a published puzzle, numbered from 1 in its collection, its grid 81 characters row by row with `.` for an empty cell |
| `mapper/sudoku_collection_mapper.py` | `SudokuCollectionMapper`: reads a collection's text into puzzles, from one 81-character line per puzzle (Norvig) or a `Grid NN` line and 9 rows (Project Euler), with `.` or `0` for an empty cell; anything else raises `ValueError` |
| `repository/sudoku_puzzle_repository.py` | `SudokuPuzzleRepository`: lists the `<collection>.txt` files of a directory and loads a collection's puzzles |

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

Without gravity, the action `place(row, col)`, with row in 1..height and col in 1..width, is legal when all of these
hold, checked in this order:

1. `payoff(X)` is unset.
2. `payoff(O)` is unset.
3. `cell(row,col)` is empty.

With gravity, the action `drop(col)`, with col in 1..width, is legal when `payoff(X)` and `payoff(O)` are unset and
`cell(1,col)`, the column's top cell, is empty.

### Transitions (predictor)

The action has one branch with probability 1. Its effects apply in order:

1. **Mark:** the landing cell gets `turn`. Without gravity it is `cell(row,col)`; with gravity, the lowest empty cell of
   column col, tried from row `height` upward.
2. **Win:** only the lines of `line` cells through the landing cell are checked: across, down and along both
   diagonals, within the grid. When the other cells of one of them all hold `turn`, `payoff(turn)` = 1 and the other
   player's payoff = 0.
3. **Draw:** when `payoff(X)` is unset and the board is full, `payoff(X)` = 0.5 and `payoff(O)` = 0.5. Without gravity
   every cell is checked; with gravity, the top row alone tells.
4. **Turn:** when `turn` is X, `turn` = O; otherwise `turn` = X.

Every landing cell carries its own win check, so a move in 4 in a row checks at most 13 lines rather than all 69.

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

The action `fill` has one parameter per empty cell, named like the cell (`cell(1,3)`), with domain 1..9. It is legal
when:

1. `payoff` is unset.
2. For each of the 27 rows, columns and boxes, `all_different` holds over its cells: the parameters of its empty cells
   and the state variables of its clues.

### Transitions (predictor)

`fill` has one branch with probability 1: it writes every parameter into its cell, then sets `payoff` = 1.0, the
share of cells filled.

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
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

names = VariableNameMapper()
interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
domain = create_domain("tictactoe")
actions = create_solver().solve(domain.problem, domain.initial_state)
# 9 actions; actions[0] is Action(name='place', parameters=(('col', 1), ('row', 1)))
predictor = Predictor(interpreter, names, expression_text, action_text)
distribution = predictor.predict(domain.transitions, domain.initial_state, actions[0])
# OutcomeDistribution(outcomes=((State(variables=(('cell(1,1)', 'X'), ('cell(1,2)', None), …, ('turn', 'O'))), 1.0),))
```

## Notes

- Tests: `builder/agent_builder_tests.py`, `builder/domain_builder_tests.py`, `factory/agent_factory_tests.py`,
  `factory/domain_factory_tests.py`, `factory/sudoku_factory_tests.py`, `factory/tictactoe_factory_tests.py`,
  `mapper/sudoku_collection_mapper_tests.py`, `repository/sudoku_puzzle_repository_tests.py`,
  `service/agent_tests.py`, `service/random_policy_tests.py`; integration:
  `test/integration/tictactoe_actions_tests.py`, `test/integration/tictactoe_search_tests.py`,
  `test/integration/tictactoe_transitions_tests.py`, `test/integration/tictactoe_variants_games_tests.py`,
  `test/integration/tictactoe_fourinarow_transitions_tests.py`, `test/integration/tictactoe_fourinarow_search_tests.py`,
  `test/integration/sudoku_solve_tests.py`, `test/integration/sudoku_collections_solve_tests.py`; end-to-end:
  `test/end_to_end/play_tictactoe_tests.py`, `test/end_to_end/play_fourinarow_tests.py`,
  `test/end_to_end/solve_sudoku_tests.py`.
