import logging
from collections.abc import Mapping

from openmind.structure.model.grid import Grid
from openmind.structure.model.scalar import Scalar
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: How a reading of one parameter is named: what a model holds where that parameter points.
AT = "{model} at {parameter}"

#: How a reading of two parameters is named: how the second stands to the first.
ROWS = "rows from {first} to {second}"
COLUMNS = "columns from {first} to {second}"
STRAIGHT = "{first} and {second} share a row or a column"
DIAGONAL = "{first} and {second} are on a diagonal"
DISTANCE = "steps from {first} to {second}"
BETWEEN = "things between {first} and {second}"
SAME = "{first} is {second}"


class ActionReadings:
    """What can be read about an action, as opposed to about a position.

    Every reading OMF had was about a state: what it holds, how much of it, what a player can do. None of them can
    say anything about a candidate action — what its source holds, how far its target is — so nothing OMF infers can
    be about how an action works. These are the readings that can.

    A parameter pointing at a cell of a grid is what makes them possible: what each grid holds there, and, for two
    such parameters, how one stands to the other. Nothing here is about any game: a parameter that doesn't name a
    cell is read as the value it is.

    The position's own scalars are read alongside, since what makes an action legal is often how it stands to one of
    them — whose turn it is, what phase the game is in."""

    def of(self, state: State, action: Action) -> dict[str, Value]:
        """Every reading of that action in that position, by name."""
        parameters = dict(action.parameters)
        grids = self._grids(state)
        readings_of_state = {name: model.value for name, model in state.models if isinstance(model, Scalar)}
        cells = {name: self._cell(grids, value) for name, value in parameters.items()}
        readings: dict[str, Value] = dict(readings_of_state)
        for name, value in parameters.items():
            at = cells[name]
            if at is None:
                readings[name] = value
                continue
            for model, grid in grids.items():
                readings[AT.format(model=model, parameter=name)] = grid.at(at) if grid.inside(at) else None
        for first, second in self._pairs(cells):
            readings.update(self._between(grids, first, second, cells[first], cells[second]))  # type: ignore[arg-type]
        return readings

    def _pairs(self, cells: Mapping[str, tuple[int, ...] | None]) -> list[tuple[str, str]]:
        named = sorted(name for name, at in cells.items() if at is not None)
        return [(first, second) for number, first in enumerate(named) for second in named[number + 1 :]]

    def _between(
        self, grids: Mapping[str, Grid], first: str, second: str, one: tuple[int, ...], other: tuple[int, ...]
    ) -> dict[str, Value]:
        """How one cell stands to another: the step between them, whether they line up, and what stands in the way."""
        if len(one) != len(other) or len(one) != 2:
            return {SAME.format(first=first, second=second): one == other}
        rows, columns = other[0] - one[0], other[1] - one[1]
        grid = next(iter(grids.values()))
        crossed = self._crossed(grid, one, other)
        return {
            ROWS.format(first=first, second=second): rows,
            COLUMNS.format(first=first, second=second): columns,
            STRAIGHT.format(first=first, second=second): (rows == 0) != (columns == 0),
            DIAGONAL.format(first=first, second=second): rows != 0 and abs(rows) == abs(columns),
            DISTANCE.format(first=first, second=second): max(abs(rows), abs(columns)),
            SAME.format(first=first, second=second): one == other,
            BETWEEN.format(first=first, second=second): sum(
                1 for model in grids.values() for cell in crossed if model.at(cell) is not None
            ),
        }

    def _crossed(self, grid: Grid, one: tuple[int, ...], other: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
        """The cells strictly between the two along the line they share; none where they share none."""
        rows, columns = other[0] - one[0], other[1] - one[1]
        if rows != 0 and columns != 0 and abs(rows) != abs(columns):
            return ()
        steps = max(abs(rows), abs(columns))
        if steps < 2:
            return ()
        row_step, column_step = (rows > 0) - (rows < 0), (columns > 0) - (columns < 0)
        return tuple((one[0] + row_step * step, one[1] + column_step * step) for step in range(1, steps))

    def _grids(self, state: State) -> dict[str, Grid]:
        return {name: model for name, model in state.models if isinstance(model, Grid)}

    def _cell(self, grids: Mapping[str, Grid], value: Value) -> tuple[int, ...] | None:
        """The cell that value points at, where it names one, and None where it names nothing: a parameter is a cell
        when a grid knows a cell by that name."""
        for grid in grids.values():
            if grid.aliases is None or not isinstance(value, str):
                continue
            try:
                return grid.aliases.to_coordinates(value)
            except (ValueError, KeyError):
                continue
        return None
