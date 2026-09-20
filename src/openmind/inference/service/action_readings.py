import logging
from collections.abc import Mapping

from openmind.structure.model.grid import Grid
from openmind.structure.model.scalar import Scalar
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.inference.model.reaching import Reaching
from openmind.inference.model.sides import Sides

logger = logging.getLogger(__name__)

#: How a reading of one parameter is named: what a model holds where that parameter points.
AT = "{model} at {parameter}"

#: How a reading of the position the action leads to is named.
AFTER = "after it, {reading}"

#: What a player can reach, where something of a kind stands.
REACHED = "{player} can reach {model} {value!r}"

#: What a player can reach, taking everything that stands there together: a white king, not a king and something white.
REACHED_HOLDING = "{player} can reach {holding}"

#: Who a reach is read for, besides each player by name: whoever is acting, and whoever is not.
ACTING = "the one acting"
ANOTHER = "another player"

#: What a player can reach where the thing standing there belongs to someone: a piece of the one acting's, rather
#: than a piece of a colour that happens to be theirs this time.
REACHED_OWNED = "{player} can reach {whose} {holding}"
#: Whose the thing standing there is, where a reach is read by role.
ACTING_OWNS, ANOTHER_OWNS, NOBODY_OWNS = "the one acting's", "another player's", "nobody's"

#: Whether a player can reach the very square an action starts from or lands on. A square holding a piece is
#: reached by a move that takes it, so this is what being attacked is: the others have a move that would capture it.
REACHES = "{player} can reach {parameter}"

#: Where a cell a parameter points at stands in the grid.
ROW = "row of {parameter}"
COLUMN = "column of {parameter}"

#: Whose what stands there is, said as a role rather than as a colour.
BELONGS = "{model} at {parameter} is {whose}"

#: Where a cell stands counted from the acting player's own end, and how far the action goes that way.
ROW_FROM_SIDE = "row of {parameter}, from the player's own side"
FORWARD = "rows from {first} to {second}, forward"

#: How a reading of two parameters is named: how the second stands to the first.
ROWS = "rows from {first} to {second}"
COLUMNS = "columns from {first} to {second}"

#: How far apart two cells are, whichever way round they stand.
ROWS_APART = "rows apart, {first} and {second}"
COLUMNS_APART = "columns apart, {first} and {second}"
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

    A parameter pointing at a cell of a grid is what makes them possible: what each grid holds there, where it
    stands, and, for two such parameters, how one stands to the other. Nothing here is about any game: a parameter
    that doesn't name a cell is read as the value it is.

    How one cell stands to another is read both signed and apart. Signed, a move of one row up and two across is a
    different fact from one row down and two across, and a piece that moves in eight directions takes eight rules
    to say so — while apart, it takes one. Signed says which way, apart says how far, and a rule usually wants only
    one of them.

    The position's own scalars are read alongside, since what makes an action legal is often how it stands to one of
    them — whose turn it is, what phase the game is in."""

    def of(
        self,
        state: State,
        action: Action,
        outcome: State | None = None,
        reach: Reaching | None = None,
        acting: str | None = None,
        sides: Sides | None = None,
    ) -> dict[str, Value]:
        """Every reading of that action in that position, by name.

        Given the position the action leads to, what that position holds is read as well — and, given something that
        says what a player can reach there, what each player could do next. A rule that forbids an action for what
        it leads to, rather than for what it is, cannot be said any other way: leaving a king where it can be taken
        is not a property of the move.

        `acting` is whose action it is, where the caller knows. Reach is then read for whoever is acting and for
        whoever is not, besides being read for each player by name — so a rule about being left open to the others
        is one rule rather than one per player, none of which says what it means.

        `sides` is what belongs to whom and which way each player faces, deduced from the game beforehand. With it,
        what stands on a square is read as the acting player's or another's rather than as a colour, and rows are
        read forward as well as up — so a pawn's step is one rule for both sides instead of two that each name a
        colour, and the ranks a pawn starts and promotes on can be spoken of at all."""
        parameters = dict(action.parameters)
        grids = self._grids(state)
        readings_of_state = {name: model.value for name, model in state.models if isinstance(model, Scalar)}
        cells = {name: self._cell(grids, value) for name, value in parameters.items()}
        readings: dict[str, Value] = dict(readings_of_state)
        for name, value in parameters.items():
            readings[name] = value
            at = cells[name]
            if at is None:
                continue
            for model, grid in grids.items():
                held = grid.at(at) if grid.inside(at) else None
                readings[AT.format(model=model, parameter=name)] = held
                if sides is not None and acting is not None:
                    readings[BELONGS.format(model=model, parameter=name, whose=ACTING)] = (
                        sides.whose(model, held) == acting
                    )
                    readings[BELONGS.format(model=model, parameter=name, whose=ANOTHER)] = sides.whose(
                        model, held
                    ) not in (acting, None)
            if len(at) == 2:
                readings[ROW.format(parameter=name)] = at[0]
                readings[COLUMN.format(parameter=name)] = at[1]
                if sides is not None and acting is not None and sides.toward(acting):
                    rows = self._rows_of(grids)
                    readings[ROW_FROM_SIDE.format(parameter=name)] = (
                        at[0] if sides.toward(acting) > 0 else rows - 1 - at[0]
                    )
        for first, second in self._pairs(cells):
            readings.update(self._between(grids, first, second, cells[first], cells[second]))  # type: ignore[arg-type]
            if sides is not None and acting is not None and sides.toward(acting):
                readings[FORWARD.format(first=first, second=second)] = (
                    readings[ROWS.format(first=first, second=second)] * sides.toward(acting)  # type: ignore[operator]
                )
        if reach is not None:
            readings.update(self._reaching(state, reach, cells, acting, sides))
        if outcome is not None:
            readings.update(self._after(outcome, reach, cells, acting, sides))
        return readings

    def _after(
        self,
        outcome: State,
        reach: "Reaching | None",
        cells: Mapping[str, tuple[int, ...] | None],
        acting: str | None,
        sides: "Sides | None" = None,
    ) -> dict[str, Value]:
        """What the position the action leads to holds, and what each player can reach in it."""
        readings: dict[str, Value] = {
            AFTER.format(reading=name): model.value for name, model in outcome.models if isinstance(model, Scalar)
        }
        grids = self._grids(outcome)
        for name, at in cells.items():
            if at is None:
                continue
            for model, grid in grids.items():
                held = grid.at(at) if grid.inside(at) else None
                readings[AFTER.format(reading=AT.format(model=model, parameter=name))] = held
                if sides is not None and acting is not None:
                    for role, whose in ((ACTING, acting), (ANOTHER, None)):
                        owner = sides.whose(model, held)
                        readings[AFTER.format(reading=BELONGS.format(model=model, parameter=name, whose=role))] = (
                            owner == whose if whose is not None else owner not in (acting, None)
                        )
        if reach is None:
            return readings
        for name, value in self._reaching(outcome, reach, cells, acting, sides).items():
            readings[AFTER.format(reading=name)] = value
        return readings

    def _reaching(
        self,
        state: State,
        reach: Reaching,
        cells: Mapping[str, tuple[int, ...] | None],
        acting: str | None,
        sides: "Sides | None",
    ) -> dict[str, Value]:
        """What each player can reach in that position: what kind of thing, whose it is, and whether it is one of
        the action's own squares.

        A square holding something is reached by a move that takes what stands there, so a player reaching the
        square an action starts from is a player who could capture the piece being moved — which is what a piece
        being attacked means, and what makes it worth moving."""
        readings: dict[str, Value] = {}
        grids = self._grids(state)
        players = list(reach.players(state))
        reached = {player: set(reach.cells(state, player)) for player in players}
        whose: list[tuple[str, list[str]]] = [(player, [player]) for player in players]
        if acting is not None and acting in reached:
            whose.append((ACTING, [acting]))
            whose.append((ANOTHER, [player for player in players if player != acting]))
        for role, held in whose:
            landing = {cell for player in held for cell in reached[player]}
            for model, grid in grids.items():
                for value in {grid.at(cell) for cell in grid.coordinates() if grid.at(cell) is not None}:
                    readings[REACHED.format(player=role, model=model, value=value)] = any(
                        grid.at(cell) == value for cell in landing
                    )
            for holding in self._holdings(grids):
                standing = any(self._holds_at(grids, cell) == holding for cell in landing)
                readings[REACHED_HOLDING.format(player=role, holding=self._said(holding))] = standing
                owner = self._owner(holding, acting, sides)
                if owner is not None:
                    name = REACHED_OWNED.format(player=role, whose=owner, holding=self._said(self._rest(holding, sides)))
                    readings[name] = readings.get(name, False) or standing
            for name, at in cells.items():
                if at is not None:
                    readings[REACHES.format(player=role, parameter=name)] = at in landing
        return readings

    def _owner(
        self, holding: tuple[tuple[str, Value], ...], acting: str | None, sides: "Sides | None"
    ) -> str | None:
        """Whose the thing standing there is, said as a role. Nothing where nothing says who owns what."""
        if sides is None or acting is None:
            return None
        for model, value in holding:
            player = sides.whose(model, value)
            if player is not None:
                return ACTING_OWNS if player == acting else ANOTHER_OWNS
        return NOBODY_OWNS

    def _rest(
        self, holding: tuple[tuple[str, Value], ...], sides: "Sides | None"
    ) -> tuple[tuple[str, Value], ...]:
        """What stands there besides who it belongs to: the kind of thing, once ownership has been said."""
        if sides is None:
            return holding
        return tuple((model, value) for model, value in holding if sides.whose(model, value) is None)

    def _holdings(self, grids: Mapping[str, Grid]) -> tuple[tuple[tuple[str, Value], ...], ...]:
        """Everything that stands anywhere, each taken as the whole of what stands there."""
        if not grids:
            return ()
        one = next(iter(grids.values()))
        found = {self._holds_at(grids, cell) for cell in one.coordinates()}
        return tuple(sorted((holding for holding in found if any(value is not None for _, value in holding)), key=repr))

    def _holds_at(self, grids: Mapping[str, Grid], cell: tuple[int, ...]) -> tuple[tuple[str, Value], ...]:
        return tuple((model, grid.at(cell) if grid.inside(cell) else None) for model, grid in sorted(grids.items()))

    def _said(self, holding: tuple[tuple[str, Value], ...]) -> str:
        return " and ".join(f"{model} {value!r}" for model, value in holding)

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
            ROWS_APART.format(first=first, second=second): abs(rows),
            COLUMNS_APART.format(first=first, second=second): abs(columns),
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

    def _rows_of(self, grids: Mapping[str, Grid]) -> int:
        """How many rows the board has, from any grid of it."""
        return next(iter(grids.values())).shape[0] if grids else 0

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
