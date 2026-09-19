# world

## Purpose

The shared vocabulary every other domain uses to describe what the agent works on: states, made of the data models of
`structure`, and actions. It depends only on `structure`.

## Content

| File | What it is |
|---|---|
| `model/state.py` | `State(models)`: named data models (see `structure/README.md`) sorted by name; `State.of(**models)`, a plain value becoming a Scalar; `model(name)`, `has(name)`, `value(name)` (a scalar's value), `names()`, `with_model(name, model)` |
| `model/action.py` | `Action`: the thing performed, with its parameters sorted by name |
| `model/players.py` | `Players(names, to_act, payoff)`: the players; the model saying who acts, a Scalar naming the player to act or, where players act at once, a Map flagging each player acting; and the Map of payoffs by player |
| `model/joint_action.py` | `JointAction(actions)`: the actions players take at once, each player's name with its action, in the order of the players' names |
| `constant/players_constant.py` | `PLAYER` (`"player"`): the name a domain's rules read the player they are solved or run for by, where players act at once |
| `builder/state_builder.py` | `StateBuilder`: `with_model(name, model)` collects models into a `State`; rejects a name set twice |
| `mapper/action_text_mapper.py` | `ActionTextMapper`: readable text for logs, such as `place(col=3, row=2)`; `joint_text(joint)` for actions taken at once, `A: throw(shape='rock'), B: throw(shape='paper')` |
| `mapper/state_text_mapper.py` | `StateTextMapper`: readable text, one `name = model` line per model, a scalar as its value |
| `mapper/grid_text_mapper.py` | `GridTextMapper`: readable text with every two-dimensional grid laid out, then one `name = value` line per other model |
| `constant/grid_text_constant.py` | The grid's marks for an empty cell (`.`) and a missing cell (a blank) |
| `service/state_reader.py` | `StateReader`: a scalar's value by name, whether players act at once (`acts_at_once`), the players to act (`players_to_act`, `player_to_act`), and the payoffs in the order of the players (`payoffs`) |

## Usage

```python
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.mapper.grid_text_mapper import GridTextMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

state = State.of(cell=Grid.filled((3, 3), None).placed((1, 1), "X"), turn="O", payoff=Map.of({"X": None, "O": None}))
action = Action("place", (("col", 3), ("row", 2)))       # parameters sorted by name
```

`GridTextMapper().to_text(state)` draws the position as:

```
cell 1 2 3
   1 X . .
   2 . . .
   3 . . .
payoff = Map(items=(('O', None), ('X', None)))
turn = 'O'
```

Columns widen to fit the longest number or value, and `None` shows as `.`.

## Players acting at once

In most domains one player acts at a time, named by the `to_act` scalar (`turn = 'X'`). Where players act at once, such
as in rock paper scissors, `to_act` names a Map flagging each player acting: `turn = {A: True, B: True}` when both act,
every flag false when the game is over. Services that only handle one player to act at a time raise `ValueError`
through `player_to_act` in such a state.

## Notes

- States and actions are frozen and hashable.
- `State.of` and `StateBuilder` sort models by name. When creating an `Action` directly, sort its parameters by name.
- Tests: `model/state_tests.py`, `builder/state_builder_tests.py`, `mapper/action_text_mapper_tests.py`, `mapper/state_text_mapper_tests.py`, `mapper/grid_text_mapper_tests.py`, `service/state_reader_tests.py`.
