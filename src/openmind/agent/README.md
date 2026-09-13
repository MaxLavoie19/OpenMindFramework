# agent

## Purpose

The agent and the domains it works on. Tic-tac-toe is a domain within the agent, not a domain in the code: it exists
only as this domain's factory recipes and their constants. So far this domain holds only the tic-tac-toe factory; the
agent itself arrives with MCTS.

## Content

| File | What it is |
|---|---|
| `constant/tictactoe_constant.py` | Board size, players, empty and unset values, variable and action names |
| `factory/tictactoe_factory.py` | `create_tictactoe_initial_state()` and `create_tictactoe_problem()` |

## Tic-tac-toe

State variables:

| Variable | Values | Initial |
|---|---|---|
| `cell(row,col)`, row and col in 1..3 | `"X"`, `"O"`, or `None` when empty | `None` |
| `turn` | `"X"` or `"O"` | `"X"` |
| `payoff(X)`, `payoff(O)` | a number once the game has ended, `None` until then | `None` |

The action `place(row, col)`, with row and col in 1..3, is legal when all of these hold, checked in this order:

1. `payoff(X)` is unset.
2. `payoff(O)` is unset.
3. `cell(row,col)` is empty.

Transitions (marking the cell, passing the turn, ending the game and setting payoffs) come with the predictor.

## Usage

```python
from openmind.agent.factory.tictactoe_factory import create_tictactoe_initial_state, create_tictactoe_problem
from openmind.csp.service.solver import Solver
from openmind.expression.service.interpreter import Interpreter
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

solver = Solver(Interpreter(VariableNameMapper()))
solver.solve(create_tictactoe_problem(), create_tictactoe_initial_state())   # the 9 place actions
```

## Notes

- Tests: `factory/tictactoe_factory_tests.py`; integration: `test/integration/tictactoe_actions_tests.py`.
