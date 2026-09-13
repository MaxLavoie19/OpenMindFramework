# expression

## Purpose

Rules are structured objects, never functions or code. This domain defines the expressions rules are made of and the
interpreter that evaluates them against a state and an action. The CSP uses them for constraints; the predictor and
the strategy RBS will use them too.

## Content

| File | What it is |
|---|---|
| `model/constant.py` | `Constant(value)`: a fixed value |
| `model/state_variable.py` | `StateVariable(base, indices)`: a state variable's value; the indices complete its name |
| `model/action_parameter.py` | `ActionParameter(name)`: the value of one of the action's parameters |
| `model/equals.py` | `Equals(left, right)`: true when both sides have the same value |
| `model/not_.py` | `Not(operand)`: true when the operand is false (`not` is a Python keyword, hence `not_.py`) |
| `model/all_of.py` | `AllOf(operands)`: true when every operand is true |
| `model/any_of.py` | `AnyOf(operands)`: true when at least one operand is true |
| `model/expression.py` | `Expression`: the union of the types above |
| `mapper/expression_text_mapper.py` | `ExpressionTextMapper`: readable text for logs, such as `all(payoff(X) == None, cell(row,col) == None)` |
| `service/interpreter.py` | `Interpreter`: evaluates an expression against a state and an action |

## Usage

```python
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.expression.service.interpreter import Interpreter
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

cell_is_empty = Equals(
    StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))),   # reads cell(row,col)
    Constant(None),
)
state = State((("cell(2,3)", "X"),))
action = Action("place", (("col", 3), ("row", 2)))

Interpreter(VariableNameMapper()).evaluate(cell_is_empty, state, action)   # False
```

## Notes

- The vocabulary is closed: the interpreter evaluates only the types above and raises `TypeError` for anything else.
- An unknown state variable or action parameter raises `KeyError`.
- Adding an expression type means adding its model, adding it to `Expression`, and handling it in `Interpreter`.
- The interpreter doesn't log; the services that use it log their decisions.
- Tests: `mapper/expression_text_mapper_tests.py`, `service/interpreter_tests.py`.
