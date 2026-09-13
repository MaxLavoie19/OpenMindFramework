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
| `factory/domain_factory.py` | `create_domain(name)`: creates a domain from its name (`"tictactoe"` or `"sudoku"`) |
| `service/agent.py` | `Agent`: searches a domain's state with MCTS, guided by a rater when built with one; `search` gives the whole result, `choose` the action |
| `model/policy.py` | `Policy`: anything with `choose(domain, state) -> Action`; `Agent` and `RandomPolicy` are policies |
| `service/random_policy.py` | `RandomPolicy`: chooses uniformly among the legal actions; a baseline opponent |
| `builder/agent_builder.py` | `AgentBuilder`: sets iterations, exploration, seed and guidance (`with_guidance(rater)`), and wires the services the agent searches with; rejects missing settings and fewer than 1 iteration |
| `factory/agent_factory.py` | `create_agent(iterations=1000, seed=None)`: an agent searching with the exploration weight √2 |
| `constant/agent_constant.py` | Default iterations (1000), exploration weight (√2), and the guidance's prior weight (1.0) and rollout temperature (0.2) |
| `constant/tictactoe_constant.py` | Domain name, board size, players, empty and unset values, payoff values, variable and action names |
| `factory/tictactoe_factory.py` | `create_tictactoe_domain()`, assembled from `create_tictactoe_initial_state()`, `create_tictactoe_problem()`, `create_tictactoe_transitions()` and `create_tictactoe_players()` |
| `constant/sudoku_constant.py` | Domain name and the separator of puzzle names, box and grid size, digits, the puzzle, empty and clue marks, empty and unset values, the collection file suffix and Project Euler's format marks, payoff values, variable and action names |
| `factory/sudoku_factory.py` | `create_sudoku_domain(name="sudoku", grid=PUZZLE)`, assembled from `create_sudoku_initial_state(grid)`, `create_sudoku_problem(grid)`, `create_sudoku_transitions(grid)` and `create_sudoku_players()` |
| `model/sudoku_puzzle.py` | `SudokuPuzzle(collection, number, grid)`: a published puzzle, numbered from 1 in its collection, its grid 81 characters row by row with `.` for an empty cell |
| `mapper/sudoku_collection_mapper.py` | `SudokuCollectionMapper`: reads a collection's text into puzzles, from one 81-character line per puzzle (Norvig) or a `Grid NN` line and 9 rows (Project Euler), with `.` or `0` for an empty cell; anything else raises `ValueError` |
| `repository/sudoku_puzzle_repository.py` | `SudokuPuzzleRepository`: lists the `<collection>.txt` files of a directory and loads a collection's puzzles |

## Tic-tac-toe

### Players

`Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)"))`: `turn` names the player to act, and `payoff(X)` and
`payoff(O)` hold the players' payoffs.

### State variables

| Variable | Values | Initial |
|---|---|---|
| `cell(row,col)`, row and col in 1..3 | `"X"`, `"O"`, or `None` when empty | `None` |
| `turn` | `"X"` or `"O"` | `"X"` |
| `payoff(X)`, `payoff(O)` | 1 for a win, 0 for a loss, 0.5 each for a draw; `None` until the game ends | `None` |

### Constraints (CSP)

The action `place(row, col)`, with row and col in 1..3, is legal when all of these hold, checked in this order:

1. `payoff(X)` is unset.
2. `payoff(O)` is unset.
3. `cell(row,col)` is empty.

### Transitions (predictor)

`place(row, col)` has one branch with probability 1. Its effects apply in order:

1. `cell(row,col)` = `turn`.
2. For each player P, when one of the 8 lines (3 rows, 3 columns, 2 diagonals) is all P: `payoff(P)` = 1 and the
   other player's payoff = 0.
3. When `payoff(X)` is unset and no cell is empty: `payoff(X)` = 0.5 and `payoff(O)` = 0.5.
4. When `turn` is X, `turn` = O; otherwise `turn` = X.

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
  `test/integration/tictactoe_transitions_tests.py`, `test/integration/sudoku_solve_tests.py`,
  `test/integration/sudoku_collections_solve_tests.py`; end-to-end:
  `test/end_to_end/play_tictactoe_tests.py`, `test/end_to_end/solve_sudoku_tests.py`.
