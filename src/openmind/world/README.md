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
| `model/players.py` | `Players(names, to_act, payoffs)`: the players, the variable naming the player to act (or, where players act at once, the base of one flag per player, `turn(A)`, `turn(B)`), and each player's payoff variable |
| `model/joint_action.py` | `JointAction(actions)`: the actions players take at once, each player's name with its action, in the order of the players' names |
| `constant/players_constant.py` | `PLAYER` (`"player"`): the name a domain's rules read the player they are solved or run for by, where players act at once |
| `builder/state_builder.py` | `StateBuilder`: collects variables into a `State`; rejects a name set twice |
| `mapper/variable_name_mapper.py` | `VariableNameMapper`: builds indexed variable names such as `cell(2,3)` and splits them back into base and indices |
| `mapper/action_text_mapper.py` | `ActionTextMapper`: readable text for logs, such as `place(col=3, row=2)`; `joint_text(joint)` for actions taken at once, `A: throw(shape='rock'), B: throw(shape='paper')` |
| `mapper/state_text_mapper.py` | `StateTextMapper`: readable text, one `name = value` line per variable |
| `mapper/grid_text_mapper.py` | `GridTextMapper`: readable text with each `<base>(<row>,<col>)` family of variables drawn as a grid, then one `name = value` line per other variable; `openmind-play` prints states with it |
| `constant/grid_text_constant.py` | The grid's marks for an empty cell (`.`) and a missing cell (a blank) |
| `service/state_reader.py` | `StateReader`: reads a variable's value by name (an unknown name raises `KeyError`), whether players act at once (`acts_at_once`), the indices of the players to act (`players_to_act`), the index of the one player to act (`player_to_act`, raising `ValueError` where several act at once), and the payoffs (a payoff that isn't a number raises `ValueError`) |

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

`GridTextMapper(VariableNameMapper()).to_text(state)` draws a tic-tac-toe position as:

```
cell 1 2 3
   1 X . .
   2 . O .
   3 . . .
payoff(O) = None
payoff(X) = None
turn = 'X'
```

A grid spans its lowest to highest row and column numbers, under the base name and the column numbers; columns widen
to fit the longest number or value, `None` shows as `.`, and a cell without a variable is left blank.

## Players acting at once

In most domains one player acts at a time, named by the `to_act` variable (`turn = 'X'`). In a domain where players
act at once, such as rock paper scissors, the state has no `to_act` variable but one flag per player under that base:
`turn(A) = True` and `turn(B) = True` when both act, every flag false when the game is over. State values can't hold
several names, so flags stand in for a list. Services that only handle one player to act at a time raise `ValueError`
through `player_to_act` in such a state.

## Notes

- States and actions are frozen and hashable.
- `StateBuilder` sorts variables by name. When creating an `Action` directly, sort its parameters by name.
- Tests: `builder/state_builder_tests.py`, `mapper/variable_name_mapper_tests.py`, `mapper/action_text_mapper_tests.py`, `mapper/state_text_mapper_tests.py`, `mapper/grid_text_mapper_tests.py`, `service/state_reader_tests.py`.
