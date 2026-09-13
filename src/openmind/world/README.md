# world

## Purpose

The shared vocabulary every other domain uses to describe what the agent works on: states, actions and the values
they hold. It depends on no other domain.

## Content

| File | What it is |
|---|---|
| `model/value.py` | `Value`: a variable or parameter value (`str`, `int`, `float`, `bool` or `None`) |
| `model/state.py` | `State`: named variables with their values, sorted by name |
| `model/action.py` | `Action`: the thing performed, with its parameters sorted by name |
| `model/players.py` | `Players(names, to_act, payoffs)`: the players, the variable naming the player to act, and each player's payoff variable |
| `builder/state_builder.py` | `StateBuilder`: collects variables into a `State`; rejects a name set twice |
| `mapper/variable_name_mapper.py` | `VariableNameMapper`: builds indexed variable names such as `cell(2,3)` |
| `mapper/action_text_mapper.py` | `ActionTextMapper`: readable text for logs, such as `place(col=3, row=2)` |
| `mapper/state_text_mapper.py` | `StateTextMapper`: readable text, one `name = value` line per variable |
| `service/state_reader.py` | `StateReader`: reads a variable's value by name (an unknown name raises `KeyError`), the index of the player to act, and the payoffs (a payoff that isn't a number raises `ValueError`) |

## Usage

```python
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.action import Action

names = VariableNameMapper()
state = (
    StateBuilder()
    .with_variable(names.to_name("cell", (2, 3)), "X")   # "cell(2,3)"
    .with_variable("turn", "O")
    .build()
)
# State(variables=(('cell(2,3)', 'X'), ('turn', 'O')))

action = Action("place", (("col", 3), ("row", 2)))       # parameters sorted by name
```

## Notes

- States and actions are frozen and hashable.
- `StateBuilder` sorts variables by name. When creating an `Action` directly, sort its parameters by name.
- Tests: `builder/state_builder_tests.py`, `mapper/variable_name_mapper_tests.py`, `mapper/action_text_mapper_tests.py`, `mapper/state_text_mapper_tests.py`, `service/state_reader_tests.py`.
