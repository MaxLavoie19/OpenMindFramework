# structure

## Purpose

The data models a state is made of, with OMF's methods, which rules and every other package use: `Scalar`, `List`,
`Map` and `Grid`. A state is named data models: `cell` a Grid, `turn` a Scalar, `payoff` a Map. Every model is
immutable, hashable and compared by content, so a state stays usable as a key in the search. Every change gives a new
model.

A rule reads a scalar as its value (`turn == "X"`) and every other model as itself (`cell[2, 3]`,
`cell.lines_through((2, 2), 3)`). It writes a model by assigning its name: a value to a scalar, a new model to the
others (`cell = cell.placed((row, col), turn)`, `payoff = payoff.with_item("X", WIN)`). Every rule can also build
models: `Grid.filled((3, 3), None)`, `Map.of({"X": None, "O": None})`.

A grid has any number of dimensions. In two it is read by (row, column), each counted from 1, row 1 at the top, or by
the aliases a game gives its cells, such as chess's `a1`: coordinates stay as they are, and the alias decides how its
names map onto them. Lines and rays follow the directions a game declares for its grid; OMF offers the orthogonal and
the diagonal sets, and a grid that declares none uses both.

This package depends on nothing else in OMF: `world` builds states from it.

## Content

| File | What it is |
|---|---|
| `model/value.py` | `Value`: what a model holds, `str`, `int`, `float`, `bool` or `None` |
| `model/coordinates.py` | `Coordinates`: one whole number per dimension, from 1 |
| `model/scalar.py` | `Scalar(value)`; `with_value` |
| `model/list.py` | `List(items)`: indexing, `len`, `in`; `appended`, `removed`, `replaced`, `count` |
| `model/map.py` | `Map(items)` sorted by key; `Map.of(mapping)`, `m[key]`, `get`, `keys`, `values`, `with_item` |
| `model/grid.py` | `Grid(shape, cells, aliases=None, directions=())`, cells row-major; `Grid.filled(shape, value)`; reading: `grid[coordinates]`, `at`, `coordinates`, `items`, `where`, `inside`, `alias`, `neighbours(where, diagonal_too=True)`, `line`, `lines(length)`, `lines_through(where, length)`, `rows`, `columns`, `diagonals`, `ray(start, direction, blocked)`, `distance(first, second, metric)`, `box`, `boxes`, `rotated`, `reflected`; changing: `placed`, `moved`, `removed` |
| `model/grid_aliases.py` | `GridAliases`: `to_coordinates(alias)`, `to_alias(coordinates)` |
| `model/data_model.py` | `DataModel`: `Scalar`, `List`, `Grid` or `Map` |
| `constant/direction_constant.py` | `orthogonal(dimensions)`, `diagonal(dimensions)`, `ORTHOGONAL` and `DIAGONAL` in two dimensions; the distances `CHEBYSHEV` (a king's steps) and `MANHATTAN` (a rook's steps without turning) |

## Usage

```python
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.model.state import State

state = State.of(cell=Grid.filled((3, 3), None), turn="X", payoff=Map.of({"X": None, "O": None}))
board = state.model("cell").placed((2, 2), "X")
won = any(all(board[where] == "X" for where in line) for line in board.lines(3))
```

## Logs

Nothing here logs: data models are data.
