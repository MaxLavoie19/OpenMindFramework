# Interfaces: structure

These are proposed interfaces, waiting for Maxime's OK. No code is written until then. Anything marked **Open** isn't
decided.

## Decided so far (2026-09-18)

- **State of data models.** OMF exposes data models with predefined methods: scalar, list, grid, map. A state is
  named data models, such as `cell` a Grid, `turn` a Scalar and `payoff` a Map. The flat `cell(2,3)` naming goes away.
- **This step builds** Scalar, List, Grid and Map.
- **Grid coordinates** are (row, column), with aliases for games like chess, where a square reads `a1`.
- **Grids have any number of dimensions.**
- **Actions:** grid helpers where OMF has them. Where it doesn't, OMF or the integrator provides ready-made actions.

## Where `structure` sits

The data models are what a state is made of, so `structure` sits below `world`: `world` imports `structure`, which
imports nothing from OMF. This changes the provisional package map, where `structure` came after `knowledge`.

## Models

All are immutable, hashable and compared by content, so states stay usable as keys in the search. Every change gives a
new model.

```python
DataModel = Scalar | List | Grid | Map
Coordinates = tuple[int, ...]            # (row, column) in two dimensions, from 1; any number of dimensions


@dataclass(frozen=True, slots=True)
class Scalar:
    value: Value
    def with_value(self, value: Value) -> Scalar: ...


@dataclass(frozen=True, slots=True)
class List:
    items: tuple[Value, ...]
    def appended(self, item: Value) -> List: ...
    def removed(self, index: int) -> List: ...
    def replaced(self, index: int, item: Value) -> List: ...
    def count(self, item: Value) -> int: ...


@dataclass(frozen=True, slots=True)
class Map:
    items: tuple[tuple[Value, Value], ...]  # sorted by key
    def get(self, key: Value) -> Value: ...
    def with_item(self, key: Value, value: Value) -> Map: ...
    def keys(self) -> tuple[Value, ...]: ...


@dataclass(frozen=True, slots=True)
class Grid:
    shape: tuple[int, ...]                # (rows, columns), or more dimensions
    cells: tuple[Value, ...]              # row-major
    aliases: GridAliases | None = None    # such as chess's a1 … h8
    directions: tuple[Coordinates, ...] = ()   # the game's own; ORTHOGONAL + DIAGONAL where none are declared

    # reading
    def at(self, where: Coordinates | str) -> Value: ...          # coordinates or an alias
    def coordinates(self) -> tuple[Coordinates, ...]: ...
    def where(self, value: Value) -> tuple[Coordinates, ...]: ...
    def inside(self, where: Coordinates) -> bool: ...
    def neighbours(self, where: Coordinates, diagonal: bool = True) -> tuple[Coordinates, ...]: ...
    def line(self, start: Coordinates, direction: Coordinates, length: int) -> tuple[Coordinates, ...] | None: ...
    def lines(self, length: int) -> tuple[tuple[Coordinates, ...], ...]: ...   # every line of that length, along the grid's directions
    def rows(self) -> tuple[tuple[Coordinates, ...], ...]: ...
    def columns(self) -> tuple[tuple[Coordinates, ...], ...]: ...
    def diagonals(self) -> tuple[tuple[Coordinates, ...], ...]: ...
    def ray(self, start: Coordinates, direction: Coordinates, blocked: Callable[[Value], bool]) -> tuple[Coordinates, ...]: ...
    def distance(self, first: Coordinates, second: Coordinates, metric: str = "chebyshev") -> int: ...   # chebyshev, manhattan
    def box(self, corner: Coordinates, size: tuple[int, ...]) -> tuple[Coordinates, ...]: ...
    def boxes(self, size: tuple[int, ...]) -> tuple[tuple[Coordinates, ...], ...]: ...   # sudoku's 3x3 boxes
    def rotated(self, turns: int = 1) -> Grid: ...                  # a quarter turn at a time, in the first two dimensions
    def reflected(self, dimension: int) -> Grid: ...
    # changing: helpers that effects rules call
    def placed(self, where: Coordinates | str, value: Value) -> Grid: ...
    def moved(self, source: Coordinates | str, target: Coordinates | str, empty: Value) -> Grid: ...
    def removed(self, where: Coordinates | str, empty: Value) -> Grid: ...


# ORTHOGONAL and DIAGONAL: OMF's direction sets for any number of dimensions (in 2D, the 4 orthogonal and 4 diagonal steps).


class GridAliases(Protocol):
    """How a game names cells: chess's files a–h and ranks 1–8."""
    def to_coordinates(self, alias: str) -> Coordinates: ...
    def to_alias(self, where: Coordinates) -> str: ...
```

`State` (in `world`) becomes `State(models: tuple[tuple[str, DataModel], ...])`, sorted by name, with `model(name)` and
`with_model(name, model)`.

## Open points

- **S1 (decided): switch now.** `State` switches to data models in this step, and every package is repaired in it.
- **S1, as asked: sequencing.** Every package reads states today as flat variables: the CSP, the predictor, the RBS's namespace
  for rules, the game factories, the text mappers, the inference engine's mechanics and deduction, and chess in
  OpenMindChess. Options:
  - Build the data models now, with their tests, and switch `State` over in the game step, which reworks the CSP, the
    predictor and the game runtime anyway.
  - Switch `State` now and repair every package in this step.
- **S2 (decided): declared per game.** OMF offers the orthogonal and the diagonal sets, and a game declares which
  directions its grid uses.
- **S2, as asked: directions for lines and rays.** In two dimensions: the four orthogonal and four diagonal steps. In more
  dimensions: every step of −1, 0 or 1 on each axis except all 0. Is that right, or should a game declare its own?
- **S3 (decided):** both Chebyshev and Manhattan.
- **S3, as asked: distance metrics.** Chebyshev (a king's steps) and Manhattan (a rook's steps without turning) to start. Others
  are added when a game needs one.
- **S4 (decided): the alias decides.** Coordinates stay (row, column) from the top, and a game's alias maps its names
  onto them.
- **S4, as asked: chess aliases.** Row 1 is the top, so `a8` is (1, 1) and `a1` is (8, 1). Is that the convention, or does chess
  put rank 1 at row 1?
