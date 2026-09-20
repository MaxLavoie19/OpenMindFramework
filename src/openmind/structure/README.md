# structure

## Purpose

The data models a state is made of, with OMF's methods, which rules and every other package use: `Scalar`, `List`,
`Map` and `Grid`. A state is named data models: `cell` a Grid, `turn` a Scalar, `payoff` a Map. Every model is
immutable, hashable and compared by content, so a state stays usable as a key in the search. Every change gives a new
model.

A rule reads a scalar as its value (`turn == "X"`) and every other model as itself (`cell[2, 3]`,
`cell.lines_through((2, 2), 3)`). It writes a model by assigning its name: a value to a scalar, a new model to the
others (`cell = cell.placed((row, col), turn)`, `payoff = payoff.with_item("X", WIN)`). Every rule can also build
models: `Grid.filled((3, 3), None)`, `Grid.of([["rook", "knight"], ["pawn", None]])`, `Map.of({"X": None, "O": None})`, and `CellNames`, so a state a game built from them reads back from what a game declared.

A grid has any number of dimensions. In two it is read by (row, column), each counted from 1, row 1 at the top, or by
the names a game gives its cells, such as chess's `a1`: coordinates stay as they are, and the names decide how they
map onto them. `CellNames(columns, rows)` is what a game usually names them with — chess's are
`CellNames(tuple("abcdefgh"), tuple("87654321"))`, so row 1 is rank 8 and a board read row by row is a board as white
sees it — and `GridAliases` stays the port for a game that names its cells some other way. A game populates a grid
with `Grid.of`, writing the cells out as they are laid out, and declares that state as where its game starts. Lines and rays follow the directions a game declares for its grid; OMF offers the orthogonal and
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
| `model/grid.py` | `Grid(shape, cells, aliases=None, directions=())`, cells row-major; `Grid.filled(shape, value)`, `Grid.of(cells)` from the cells written out as they are laid out; reading: `grid[coordinates]`, `at`, `coordinates`, `items`, `where`, `inside`, `alias`, `neighbours(where, diagonal_too=True)`, `line`, `lines(length)`, `lines_through(where, length)`, `rows`, `columns`, `diagonals`, `ray(start, direction, blocked)`, `distance(first, second, metric)`, `box`, `boxes`, `rotated`, `reflected`; changing: `placed`, `moved`, `removed` |
| `model/grid_aliases.py` | `GridAliases`: `to_coordinates(alias)`, `to_alias(coordinates)` |
| `model/cell_names.py` | `CellNames(columns, rows)`: the names a game gives a grid's cells, a cell's name being its column's and its row's — chess's `a1`, a spreadsheet's `B3` |
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

## Notes

- Tests: `model/grid_tests.py`, `model/cell_names_tests.py`, `model/list_tests.py`, `model/map_tests.py`,
  `model/scalar_tests.py`.
