import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant
from openmind.structure.model.grid import Grid
from openmind.structure.model.scalar import Scalar
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.inference.model.example import Example
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

#: A reading of the position the action leads to, as a predicate's prefix rather than as a wrapped name.
AFTERWARDS = "after it"

#: Each reading, and the predicate it is said as.
#:
#: **A template is how a reading is written out for a reader; the predicate is what it is.** The two were once one
#: thing — a name was made by filling a template, and the filled string was the key — and turning that back into a
#: literal meant matching the string against every template in order and taking the first that fitted. Three of
#: these are the same shape once filled: `{player} can reach {parameter}`, `{player} can reach {holding}` and
#: `{player} can reach {whose} {holding}` are each some text, " can reach ", and more text. Matched in order they
#: collapsed — of eighty-three readings of one position, fifty-two landed in two predicates, and
#: `another player can reach source` came out as `can reach holding('another player', 'source', True)`, naming a
#: square as though it were a thing standing on one. That is the reading whose own comment says it is what being
#: attacked means.
#:
#: Said outright there is nothing to match and nothing to guess: the slots are the terms, in the template's order,
#: with what was read last. Four reaches, four predicates, four arities.
SAID: tuple[tuple[str, str], ...] = (
    (AT, "at"),
    (BELONGS, "belongs"),
    (REACHED, "can reach a thing"),
    (REACHED_HOLDING, "can reach holding"),
    (REACHED_OWNED, "can reach owned"),
    (REACHES, "can reach"),
    (ROW, "row"),
    (COLUMN, "column"),
    (ROW_FROM_SIDE, "row from own side"),
    (FORWARD, "rows forward"),
    (ROWS, "rows"),
    (COLUMNS, "columns"),
    (ROWS_APART, "rows apart"),
    (COLUMNS_APART, "columns apart"),
    (STRAIGHT, "share a row or a column"),
    (DIAGONAL, "on a diagonal"),
    (DISTANCE, "steps"),
    (BETWEEN, "things between"),
    (SAME, "is"),
)

#: Each template's predicate and the slots it is said of, in the order the template writes them.
SLOTS: Mapping[str, tuple[str, tuple[str, ...]]] = {
    template: (predicate, tuple(one.removesuffix("!r") for one in re.findall(r"\{([a-z_]+(?:!r)?)\}", template)))
    for template, predicate in SAID
}


@dataclass(frozen=True, slots=True)
class Reading:
    """One thing read of an action, before it is either said as a literal or written out as a name.

    It carries the slots apart from the template rather than formatted into it, which is the whole point: a name
    is one presentation of this and a literal is another, and neither is the reading."""

    #: The template this is a reading of, or the whole name where a game says something in its own way — a
    #: scalar of the position, a parameter of the action. Those have no slots and are one term.
    template: str
    #: What the template's slots hold, in the template's own order.
    said_of: tuple[Value, ...] = ()
    #: What was read.
    value: Value = None
    #: Whether this is read of the position the action leads to rather than of the one it is made in.
    afterwards: bool = False

    @property
    def name(self) -> str:
        """The reading written out, as the engine it was built for knows it."""
        said = self.template
        if self.template in SLOTS:
            said = self.template.format(**dict(zip(SLOTS[self.template][1], self.said_of, strict=True)))
        return AFTER.format(reading=said) if self.afterwards else said

    @property
    def literal(self) -> Literal:
        """The reading said outright, its slots become terms and what was read the last of them."""
        predicate = SLOTS[self.template][0] if self.template in SLOTS else self.template
        return Literal(
            f"{AFTERWARDS}, {predicate}" if self.afterwards else predicate,
            (*(Constant(one) for one in self.said_of), Constant(self.value)),
        )


def a_reading(template: str, read: Value, /, **said_of: Value) -> Reading:
    """One reading, its slots given by name and kept in the template's order.

    The template and what was read are positional only, because one of the templates has a slot called `value`
    and a keyword here would shadow it."""
    _, slots = SLOTS[template]
    return Reading(template, tuple(said_of[slot] for slot in slots), read)


def afterwards(reading: Reading) -> Reading:
    """That reading, read of the position the action leads to."""
    return Reading(reading.template, reading.said_of, reading.value, True)


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
        """Every reading of that action in that position, by the name the stand-in engine knows it by.

        **One presentation of `read`, kept for the engine this was written before.** `RuleDeducer` and the
        extraction scripts match a reading by its filled-in name; they go when the engine that replaces them
        lands, and this goes with them. Nothing new should be written against it — `literals` is the reading
        said in a way a rule can quantify over."""
        return {one.name: one.value for one in self.read(state, action, outcome, reach, acting, sides)}

    def literals(
        self,
        state: State,
        action: Action,
        outcome: State | None = None,
        reach: Reaching | None = None,
        acting: str | None = None,
        sides: Sides | None = None,
    ) -> tuple[Literal, ...]:
        """Every reading of that action, said outright: the slots are terms and what was read is the last of them.

        **This is what makes a reading general.** `rows from source to target`, read as 3, is
        `rows(source, target, 3)` — so a clause may put a variable where the 3 is, or where the source is, or tie
        one reading's player to another's. One clause says what a pawn does for both players where a name could
        only ever be compared to a value."""
        return tuple(one.literal for one in self.read(state, action, outcome, reach, acting, sides))

    def example(self, readings: Sequence[Reading], holds: bool, where: object | None = None) -> Example:
        """One case to learn from: everything read of it, and whether the thing being learned held."""
        return Example(tuple(one.literal for one in readings), holds, where)

    def examples(
        self, seen: Sequence[tuple[Sequence[Reading], bool]], within: Sequence[object] = ()
    ) -> tuple[Example, ...]:
        """Cases from readings already gathered, each with the position it was read in where there is one."""
        places = tuple(within) if len(within) == len(seen) else (None,) * len(seen)
        return tuple(self.example(readings, holds, place) for (readings, holds), place in zip(seen, places))

    def read(
        self,
        state: State,
        action: Action,
        outcome: State | None = None,
        reach: Reaching | None = None,
        acting: str | None = None,
        sides: Sides | None = None,
    ) -> tuple[Reading, ...]:
        """Every reading of that action in that position, before it is named or said.

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
        cells = {name: self._cell(grids, value) for name, value in parameters.items()}
        readings: list[Reading] = [
            Reading(name, value=model.value) for name, model in state.models if isinstance(model, Scalar)
        ]
        for name, value in parameters.items():
            readings.append(Reading(name, value=value))
            at = cells[name]
            if at is None:
                continue
            for model, grid in grids.items():
                held = grid.at(at) if grid.inside(at) else None
                readings.append(a_reading(AT, held, model=model, parameter=name))
                if sides is not None and acting is not None:
                    readings.append(
                        a_reading(BELONGS, sides.whose(model, held) == acting, model=model, parameter=name, whose=ACTING)
                    )
                    readings.append(
                        a_reading(
                            BELONGS,
                            sides.whose(model, held) not in (acting, None),
                            model=model, parameter=name, whose=ANOTHER,
                        )
                    )
            if len(at) == 2:
                readings.append(a_reading(ROW, at[0], parameter=name))
                readings.append(a_reading(COLUMN, at[1], parameter=name))
                if sides is not None and acting is not None and sides.toward(acting):
                    rows = self._rows_of(grids)
                    readings.append(
                        a_reading(
                            ROW_FROM_SIDE, at[0] if sides.toward(acting) > 0 else rows - 1 - at[0], parameter=name
                        )
                    )
        for first, second in self._pairs(cells):
            between = self._between(grids, first, second, cells[first], cells[second])  # type: ignore[arg-type]
            readings.extend(between)
            if sides is not None and acting is not None and sides.toward(acting):
                rows = next((one.value for one in between if one.template == ROWS), None)
                if rows is not None:
                    readings.append(
                        a_reading(FORWARD, rows * sides.toward(acting), first=first, second=second)  # type: ignore[operator]
                    )
        if reach is not None:
            readings.extend(self._reaching(state, reach, cells, acting, sides))
        if outcome is not None:
            readings.extend(self._after(outcome, reach, cells, acting, sides))
        return tuple(readings)

    def _after(
        self,
        outcome: State,
        reach: "Reaching | None",
        cells: Mapping[str, tuple[int, ...] | None],
        acting: str | None,
        sides: "Sides | None" = None,
    ) -> list[Reading]:
        """What the position the action leads to holds, and what each player can reach in it."""
        readings: list[Reading] = [
            Reading(name, value=model.value, afterwards=True)
            for name, model in outcome.models
            if isinstance(model, Scalar)
        ]
        grids = self._grids(outcome)
        for name, at in cells.items():
            if at is None:
                continue
            for model, grid in grids.items():
                held = grid.at(at) if grid.inside(at) else None
                readings.append(afterwards(a_reading(AT, held, model=model, parameter=name)))
                if sides is not None and acting is not None:
                    for role, whose in ((ACTING, acting), (ANOTHER, None)):
                        owner = sides.whose(model, held)
                        readings.append(
                            afterwards(
                                a_reading(
                                    BELONGS,
                                    owner == whose if whose is not None else owner not in (acting, None),
                                    model=model, parameter=name, whose=role,
                                )
                            )
                        )
        if reach is None:
            return readings
        readings.extend(afterwards(one) for one in self._reaching(outcome, reach, cells, acting, sides))
        return readings

    def _reaching(
        self,
        state: State,
        reach: Reaching,
        cells: Mapping[str, tuple[int, ...] | None],
        acting: str | None,
        sides: "Sides | None",
    ) -> list[Reading]:
        """What each player can reach in that position: what kind of thing, whose it is, and whether it is one of
        the action's own squares.

        A square holding something is reached by a move that takes what stands there, so a player reaching the
        square an action starts from is a player who could capture the piece being moved — which is what a piece
        being attacked means, and what makes it worth moving.

        **The four of them are four predicates and were one.** Reaching a thing of a kind, reaching a whole
        holding, reaching a holding of somebody's, and reaching a square the action itself names are written
        alike — some text, " can reach ", more text — so matching a filled name against templates in order put
        them all under whichever template came first. Said outright they never meet."""
        readings: list[Reading] = []
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
                    readings.append(
                        a_reading(
                            REACHED,
                            any(grid.at(cell) == value for cell in landing),
                            player=role, model=model, value=value,
                        )
                    )
            # Several holdings answer to one owned reading once whose it is has been taken out of them, so that
            # one is true where any of them stands — gathered before it is said, since a reading said twice is
            # two readings where a name written twice was one.
            owned: dict[tuple[str, str], bool] = {}
            for holding in self._holdings(grids):
                standing = any(self._holds_at(grids, cell) == holding for cell in landing)
                readings.append(a_reading(REACHED_HOLDING, standing, player=role, holding=self._said(holding)))
                owner = self._owner(holding, acting, sides)
                if owner is not None:
                    rest = self._said(self._rest(holding, sides))
                    owned[(owner, rest)] = owned.get((owner, rest), False) or standing
            readings.extend(
                a_reading(REACHED_OWNED, standing, player=role, whose=owner, holding=rest)
                for (owner, rest), standing in owned.items()
            )
            for name, at in cells.items():
                if at is not None:
                    readings.append(a_reading(REACHES, at in landing, player=role, parameter=name))
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
    ) -> list[Reading]:
        """How one cell stands to another: the step between them, whether they line up, and what stands in the way."""
        said = lambda template, value: a_reading(template, value, first=first, second=second)  # noqa: E731
        if len(one) != len(other) or len(one) != 2:
            return [said(SAME, one == other)]
        rows, columns = other[0] - one[0], other[1] - one[1]
        grid = next(iter(grids.values()))
        crossed = self._crossed(grid, one, other)
        return [
            said(ROWS, rows),
            said(COLUMNS, columns),
            said(ROWS_APART, abs(rows)),
            said(COLUMNS_APART, abs(columns)),
            said(STRAIGHT, (rows == 0) != (columns == 0)),
            said(DIAGONAL, rows != 0 and abs(rows) == abs(columns)),
            said(DISTANCE, max(abs(rows), abs(columns))),
            said(SAME, one == other),
            said(BETWEEN, sum(1 for model in grids.values() for cell in crossed if model.at(cell) is not None)),
        ]

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
