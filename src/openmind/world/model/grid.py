import itertools
from collections.abc import Iterator

from openmind.world.model.value import Value

#: A cell's coordinates: a whole number on a grid of one dimension, a tuple of whole numbers otherwise.
type Coordinates = int | tuple[int, ...]


class Grid(dict[Coordinates, Value]):
    """The variables of a base whose indices are all whole numbers, by coordinates, searchable by those coordinates.
    Nothing here knows a game: a line is a row, a column or a diagonal, in any number of dimensions. It is a dict, so
    rules read and write it as they do any base.

    A grid remembers each cell's rays and neighbours, since a position is read by many expressions; writing to the grid
    forgets them."""

    __slots__ = ("_rays", "_neighbours")

    def __init__(self, *arguments: object, **keywords: object) -> None:
        super().__init__(*arguments, **keywords)  # type: ignore[arg-type]
        self._forget()

    def __setitem__(self, key: Coordinates, value: Value) -> None:
        super().__setitem__(key, value)
        self._forget()

    def __delitem__(self, key: Coordinates) -> None:
        super().__delitem__(key)
        self._forget()

    def update(self, *arguments: object, **keywords: object) -> None:  # type: ignore[override]
        super().update(*arguments, **keywords)  # type: ignore[arg-type]
        self._forget()

    def __reduce__(self) -> tuple[type["Grid"], tuple[dict[Coordinates, Value]]]:
        """A copy sent to another process is built from its cells, so it starts with nothing remembered."""
        return (Grid, (dict(self),))

    def where(self, value: Value) -> tuple[Coordinates, ...]:
        """The coordinates holding the value, in order."""
        return tuple(sorted(key for key, held in self.items() if held == value))  # type: ignore[type-var]

    def parity(self, at: Coordinates) -> int:
        """The sum of the coordinates, modulo 2: which of two alternating colours a cell is."""
        return (at if isinstance(at, int) else sum(at)) % 2

    def distance(self, first: Coordinates, second: Coordinates) -> int:
        """The largest gap between the coordinates: how many steps apart when a step goes in any direction."""
        return max((abs(a - b) for a, b in self._pairs(first, second)), default=0)

    def steps(self, first: Coordinates, second: Coordinates) -> int:
        """The sum of the gaps between the coordinates: how many steps apart when a step changes one coordinate."""
        return sum(abs(a - b) for a, b in self._pairs(first, second))

    def aligned(self, first: Coordinates, second: Coordinates) -> bool:
        """Whether two different cells are on one line: every coordinate that differs differs by the same amount."""
        return len({abs(a - b) for a, b in self._pairs(first, second) if a != b}) == 1

    def between(self, first: Coordinates, second: Coordinates) -> tuple[Value, ...]:
        """The values strictly between two aligned cells, from the first; none when they aren't aligned."""
        if not self.aligned(first, second):
            return ()
        start = self._coordinates(first)
        step = tuple((b > a) - (b < a) for a, b in self._pairs(first, second))
        return self._along(start, step, self.distance(first, second) - 1)

    def ray(self, at: Coordinates, direction: Coordinates) -> tuple[Value, ...]:
        """The values from the cell after `at` stepping by `direction`, until a step leaves the grid; none for no
        direction. Worked out once per cell and direction."""
        kept = self._rays.get((at, direction))
        if kept is None:
            step = self._coordinates(direction)
            kept = () if not any(step) else self._along(self._coordinates(at), step, None)
            self._rays[(at, direction)] = kept
        return kept

    def neighbours(self, at: Coordinates) -> tuple[Value, ...]:
        """The values of the cells one step away in every direction, diagonals included, in order of direction. Worked
        out once per cell."""
        kept = self._neighbours.get(at)
        if kept is None:
            start = self._coordinates(at)
            found: list[Value] = []
            for step in itertools.product((-1, 0, 1), repeat=len(start)):
                if any(step):
                    key = self._key(tuple(coordinate + delta for coordinate, delta in zip(start, step, strict=True)))
                    if key in self:
                        found.append(self[key])
            kept = self._neighbours[at] = tuple(found)
        return kept

    def _along(self, start: tuple[int, ...], step: tuple[int, ...], length: int | None) -> tuple[Value, ...]:
        """The values from the cell after `start`, stepping by `step`, for `length` cells or until the grid ends."""
        values: list[Value] = []
        coordinates = start
        while length is None or len(values) < length:
            coordinates = tuple(coordinate + delta for coordinate, delta in zip(coordinates, step, strict=True))
            key = self._key(coordinates)
            if key not in self:
                break
            values.append(self[key])
        return tuple(values)

    def _forget(self) -> None:
        self._rays: dict[tuple[Coordinates, Coordinates], tuple[Value, ...]] = {}
        self._neighbours: dict[Coordinates, tuple[Value, ...]] = {}

    def _pairs(self, first: Coordinates, second: Coordinates) -> Iterator[tuple[int, int]]:
        return zip(self._coordinates(first), self._coordinates(second), strict=True)

    def _coordinates(self, at: Coordinates) -> tuple[int, ...]:
        return (at,) if isinstance(at, int) else tuple(at)

    def _key(self, coordinates: tuple[int, ...]) -> Coordinates:
        """Coordinates as the grid keys them: a whole number on a grid of one dimension keyed by whole numbers."""
        if len(coordinates) == 1 and isinstance(next(iter(self), None), int):
            return coordinates[0]
        return coordinates

