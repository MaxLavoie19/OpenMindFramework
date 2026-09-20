from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from itertools import product
from math import prod

from openmind.structure.constant.direction_constant import CHEBYSHEV, MANHATTAN, diagonal, orthogonal
from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.grid_aliases import GridAliases
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Grid:
    """Cells in any number of dimensions, read and changed by coordinates — (row, column) in two dimensions, from 1,
    row 1 at the top — or by the aliases a game gives them, such as chess's a1. Its cells are kept row-major. Lines and
    rays follow the directions the game declared; orthogonal and diagonal steps where it declared none.

    **A cell holding None holds nothing**, which is how every game spells an empty square and what `where(None)`,
    `moved` and `removed` take it to mean. Reading a cell that isn't there raises instead, so nothing-here and
    not-a-cell are never confused. A game free to spell nothing its own way — 0 on a numeric board — passes its own
    marker where one is asked for.

    Every change gives a new grid, so states holding grids can be compared and used as keys. Rules read a cell as
    `cell[2, 3]` and write a new grid, such as `cell = cell.placed((2, 3), turn)`."""

    shape: tuple[int, ...]
    cells: tuple[Value, ...]
    aliases: GridAliases | None = field(default=None, compare=False)
    directions: tuple[Coordinates, ...] = ()

    def __post_init__(self) -> None:
        if len(self.cells) != prod(self.shape):
            raise ValueError(f"A grid of shape {self.shape} has {prod(self.shape)} cells, not {len(self.cells)}")

    @staticmethod
    def filled(
        shape: tuple[int, ...],
        value: Value,
        aliases: GridAliases | None = None,
        directions: tuple[Coordinates, ...] = (),
    ) -> "Grid":
        """Every cell holding the same value."""
        return Grid(shape, (value,) * prod(shape), aliases, directions)

    @staticmethod
    def of(
        cells: Sequence[object],
        aliases: GridAliases | None = None,
        directions: tuple[Coordinates, ...] = (),
    ) -> "Grid":
        """A grid from its cells written out, nested as it is laid out: `Grid.of([["a", "b"], ["c", "d"]])` is two rows
        of two, row 1 first, and a game writes a board as a board looks. It nests as deep as the grid has dimensions,
        so a plain sequence is one row. Rows of different lengths raise ValueError."""
        shape = _shape(cells)
        return Grid(shape, tuple(_flattened(cells, shape)), aliases, directions)

    # reading

    def __getitem__(self, where: Coordinates | str) -> Value:
        return self.at(where)

    def __iter__(self) -> Iterator[Coordinates]:
        return iter(self.coordinates())

    def at(self, where: Coordinates | str) -> Value:
        return self.cells[self._index(self._coordinates(where))]

    def coordinates(self) -> tuple[Coordinates, ...]:
        """Every cell's coordinates, row-major."""
        return tuple(product(*(range(1, size + 1) for size in self.shape)))

    def items(self) -> tuple[tuple[Coordinates, Value], ...]:
        return tuple(zip(self.coordinates(), self.cells, strict=True))

    def where(self, value: Value) -> tuple[Coordinates, ...]:
        """The coordinates holding the value, row-major."""
        return tuple(where for where, held in self.items() if held == value)

    def inside(self, where: Coordinates) -> bool:
        return len(where) == len(self.shape) and all(1 <= part <= size for part, size in zip(where, self.shape, strict=True))

    def alias(self, where: Coordinates) -> str:
        """The cell's name as the game gives it; its coordinates written out without aliases."""
        return self.aliases.to_alias(where) if self.aliases is not None else ",".join(str(part) for part in where)

    def neighbours(self, where: Coordinates, diagonal_too: bool = True) -> tuple[Coordinates, ...]:
        """The cells one step away, along one axis, or along several too."""
        dimensions = len(self.shape)
        steps = orthogonal(dimensions) + (diagonal(dimensions) if diagonal_too else ())
        return tuple(step for step in (_add(where, move) for move in steps) if self.inside(step))

    def line(self, start: Coordinates, direction: Coordinates, length: int) -> tuple[Coordinates, ...] | None:
        """The `length` cells from `start` along the direction, or None where the line leaves the grid."""
        cells = tuple(_add(start, _scale(direction, step)) for step in range(length))
        return cells if all(self.inside(cell) for cell in cells) else None

    def lines(self, length: int) -> tuple[tuple[Coordinates, ...], ...]:
        """Every line of that length along the grid's directions, each once whichever end it is read from."""
        found: dict[frozenset[Coordinates], tuple[Coordinates, ...]] = {}
        for start in self.coordinates():
            for direction in self._directions():
                line = self.line(start, direction, length)
                if line is not None:
                    found.setdefault(frozenset(line), line)
        return tuple(found.values())

    def lines_through(self, where: Coordinates, length: int) -> tuple[tuple[Coordinates, ...], ...]:
        """Every line of that length through the cell."""
        return tuple(line for line in self.lines(length) if where in line)

    def rows(self) -> tuple[tuple[Coordinates, ...], ...]:
        return tuple(tuple((row, column) for column in range(1, self.shape[1] + 1)) for row in range(1, self.shape[0] + 1))

    def columns(self) -> tuple[tuple[Coordinates, ...], ...]:
        return tuple(tuple((row, column) for row in range(1, self.shape[0] + 1)) for column in range(1, self.shape[1] + 1))

    def diagonals(self) -> tuple[tuple[Coordinates, ...], ...]:
        """Both main diagonals of a square grid."""
        size = min(self.shape[0], self.shape[1])
        return (
            tuple((index, index) for index in range(1, size + 1)),
            tuple((index, size + 1 - index) for index in range(1, size + 1)),
        )

    def ray(self, start: Coordinates, direction: Coordinates, blocked: Callable[[Value], bool]) -> tuple[Coordinates, ...]:
        """The cells from `start` along the direction, the start left out, up to and including the first blocked one,
        or to the grid's edge."""
        cells: list[Coordinates] = []
        where = _add(start, direction)
        while self.inside(where):
            cells.append(where)
            if blocked(self.at(where)):
                break
            where = _add(where, direction)
        return tuple(cells)

    def distance(self, first: Coordinates, second: Coordinates, metric: str = CHEBYSHEV) -> int:
        """Chebyshev: a king's steps; Manhattan: a rook's steps without turning."""
        gaps = [abs(one - other) for one, other in zip(first, second, strict=True)]
        if metric == CHEBYSHEV:
            return max(gaps)
        if metric == MANHATTAN:
            return sum(gaps)
        raise ValueError(f"No distance {metric!r}: the distances are {CHEBYSHEV} and {MANHATTAN}")

    def box(self, corner: Coordinates, size: tuple[int, ...]) -> tuple[Coordinates, ...]:
        """The cells of the box of that size from its top-left corner."""
        return tuple(
            tuple(start + offset for start, offset in zip(corner, offsets, strict=True))
            for offsets in product(*(range(extent) for extent in size))
        )

    def boxes(self, size: tuple[int, ...]) -> tuple[tuple[Coordinates, ...], ...]:
        """The grid cut into boxes of that size, such as sudoku's 3 by 3."""
        corners = product(*(range(1, extent + 1, step) for extent, step in zip(self.shape, size, strict=True)))
        return tuple(self.box(corner, size) for corner in corners)

    def rotated(self, turns: int = 1) -> "Grid":
        """The grid turned a quarter clockwise that many times, in its first two dimensions."""
        grid = self
        for _ in range(turns % 4):
            rows, columns = grid.shape[0], grid.shape[1]
            shape = (columns, rows, *grid.shape[2:])
            moved = {(column, rows + 1 - row, *rest): value for (row, column, *rest), value in grid.items()}
            grid = Grid(shape, tuple(moved[where] for where in _coordinates_of(shape)), grid.aliases, grid.directions)
        return grid

    def reflected(self, dimension: int = 1) -> "Grid":
        """The grid mirrored along one dimension: 0 turns it upside down, 1 left to right."""
        size = self.shape[dimension]
        moved = {
            tuple(size + 1 - part if index == dimension else part for index, part in enumerate(where)): value
            for where, value in self.items()
        }
        return Grid(self.shape, tuple(moved[where] for where in self.coordinates()), self.aliases, self.directions)

    # changing: helpers effects rules call

    def placed(self, where: Coordinates | str, value: Value) -> "Grid":
        index = self._index(self._coordinates(where))
        return Grid(self.shape, self.cells[:index] + (value,) + self.cells[index + 1 :], self.aliases, self.directions)

    def moved(self, source: Coordinates | str, target: Coordinates | str, empty: Value = None) -> "Grid":
        """The piece at `source` moved to `target`, taking whatever was there, `source` left empty. A cell holds None
        when it holds nothing; a game that spells nothing its own way passes it."""
        return self.placed(target, self.at(source)).placed(source, empty)

    def removed(self, where: Coordinates | str, empty: Value = None) -> "Grid":
        """The cell emptied: it holds nothing, which is None unless the game spells nothing its own way."""
        return self.placed(where, empty)

    def _directions(self) -> tuple[Coordinates, ...]:
        dimensions = len(self.shape)
        return self.directions or orthogonal(dimensions) + diagonal(dimensions)

    def _coordinates(self, where: Coordinates | str) -> Coordinates:
        if isinstance(where, str):
            if self.aliases is None:
                raise KeyError(f"{where!r}: this grid has no aliases")
            return self.aliases.to_coordinates(where)
        return tuple(where)

    def _index(self, where: Coordinates) -> int:
        if not self.inside(where):
            raise KeyError(f"{where} is outside a grid of shape {self.shape}")
        index = 0
        for part, size in zip(where, self.shape, strict=True):
            index = index * size + (part - 1)
        return index


def _add(first: Coordinates, second: Coordinates) -> Coordinates:
    return tuple(one + other for one, other in zip(first, second, strict=True))


def _scale(step: Coordinates, times: int) -> Coordinates:
    return tuple(part * times for part in step)


def _coordinates_of(shape: tuple[int, ...]) -> tuple[Coordinates, ...]:
    return tuple(product(*(range(1, size + 1) for size in shape)))


def _shape(cells: object) -> tuple[int, ...]:
    """The shape the nested cells lay out: their length, then the shape of the first one, down to the values. Cells of
    different lengths raise ValueError, since a grid has no ragged rows."""
    if not _nested(cells):
        return ()
    inside = tuple(_shape(cell) for cell in cells)  # type: ignore[union-attr]
    if len(set(inside)) > 1:
        raise ValueError(f"A grid has no rows of different lengths: {sorted(set(inside))}")
    return (len(cells), *inside[0]) if inside else (0,)  # type: ignore[arg-type]


def _flattened(cells: object, shape: tuple[int, ...]) -> Iterator[Value]:
    """The cells row-major, as a grid keeps them."""
    if len(shape) <= 1:
        yield from cells  # type: ignore[misc]
        return
    for row in cells:  # type: ignore[union-attr]
        yield from _flattened(row, shape[1:])


def _nested(cells: object) -> bool:
    """Whether this is a row of cells rather than one cell's value: a string is a value, not a row of letters."""
    return isinstance(cells, Sequence) and not isinstance(cells, str | bytes)
