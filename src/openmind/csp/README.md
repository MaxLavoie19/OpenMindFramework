# csp

## Purpose

A constraint satisfaction problem describes which actions a domain allows. The solver turns a problem and a state into
the legal actions, each with its parameter values. MCTS will use it to generate the actions to explore.

## Content

| File | What it is |
|---|---|
| `model/discrete_domain.py` | `DiscreteDomain(values)`: the finite values a variable can take |
| `model/variable.py` | `Variable(name, domain)`: an action parameter to solve for |
| `model/action_definition.py` | `ActionDefinition(name, variables, constraints)`: an action and the constraints that make it legal |
| `model/problem.py` | `Problem(actions)`: every action definition of a domain |
| `builder/problem_builder.py` | `ProblemBuilder`: collects action definitions; rejects a repeated action or variable name |
| `service/solver.py` | `Solver`: finds every action that satisfies all of its constraints in a state |

## Usage

```python
from openmind.csp.builder.problem_builder import ProblemBuilder
from openmind.csp.model.discrete_domain import DiscreteDomain
from openmind.csp.model.variable import Variable
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State

positions = DiscreteDomain((1, 2))
problem = (
    ProblemBuilder()
    .with_action(
        "place",
        (Variable("row", positions), Variable("col", positions)),
        (Equals(StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))), Constant(None)),),
    )
    .build()
)
state = State((("cell(1,1)", "X"), ("cell(1,2)", None), ("cell(2,1)", None), ("cell(2,2)", None)))

names = VariableNameMapper()
Solver(Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()).solve(problem, state)
# (Action(name='place', parameters=(('col', 2), ('row', 1))),
#  Action(name='place', parameters=(('col', 1), ('row', 2))),
#  Action(name='place', parameters=(('col', 2), ('row', 2))))
```

## How the solver works

- For each action definition in order, it tries every combination of variable values; the first declared variable
  changes slowest.
- Constraints are checked in order, and a candidate is rejected at the first one that is false.
- A constraint must evaluate to `True` or `False`; any other value raises `TypeError`.
- Only discrete domains exist so far. Propagation, interval domains and bound variables come with the first problem
  that needs them.

## Logs

Logger `openmind.csp.service.solver`:

- `DEBUG Accepted place(col=2, row=1)`
- `DEBUG Rejected place(col=1, row=1): cell(row,col) == None is false`
- `DEBUG 3 of 4 candidate actions are legal`

Actions and constraints are written by `ActionTextMapper` and `ExpressionTextMapper`.

## Notes

- Tests: `builder/problem_builder_tests.py`, `service/solver_tests.py`; integration:
  `test/integration/tictactoe_actions_tests.py`.
