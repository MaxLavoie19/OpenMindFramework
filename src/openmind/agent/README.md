# agent

## Purpose

The agent and the domains it works on. Tic-tac-toe is a domain within the agent, not a domain in the code: it exists
only as this domain's factory recipes and their constants. So far this domain holds only the tic-tac-toe factory; the
agent itself arrives with MCTS.

## Content

| File | What it is |
|---|---|
| `constant/tictactoe_constant.py` | Board size, players, empty and unset values, payoff values, variable and action names |
| `factory/tictactoe_factory.py` | `create_tictactoe_initial_state()`, `create_tictactoe_problem()` and `create_tictactoe_transitions()` |

## Tic-tac-toe

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

## Usage

```python
from openmind.agent.factory.tictactoe_factory import (
    create_tictactoe_initial_state,
    create_tictactoe_problem,
    create_tictactoe_transitions,
)
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper

names = VariableNameMapper()
interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
state = create_tictactoe_initial_state()
actions = Solver(interpreter, expression_text, action_text).solve(create_tictactoe_problem(), state)
# 9 actions; actions[0] is Action(name='place', parameters=(('col', 1), ('row', 1)))
predictor = Predictor(interpreter, names, expression_text, action_text)
distribution = predictor.predict(create_tictactoe_transitions(), state, actions[0])
# OutcomeDistribution(outcomes=((State(variables=(('cell(1,1)', 'X'), ('cell(1,2)', None), …, ('turn', 'O'))), 1.0),))
```

## Notes

- Tests: `factory/tictactoe_factory_tests.py`; integration: `test/integration/tictactoe_actions_tests.py`,
  `test/integration/tictactoe_transitions_tests.py`.
