import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from openmind.inference.constant.inference_constant import BEST, COUNT, HERE, ME, OTHER, OUTSIDE, WORST
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.structure.model.grid import Grid
from openmind.world.model.state import State
from openmind.structure.model.value import Value

if TYPE_CHECKING:
    from openmind.inference.service.mechanics import Mechanics

type Reading = Callable[["PositionView"], object]
type Moves = tuple[tuple[tuple["PositionView", float], ...], ...]


class PositionView:
    """A position as a generated expression reads it. Its models are attributes, read the way rules read them: a scalar
    as its value, `view.turn`; a grid, list or map as itself, `view.cell[2, 3]`, `view.payoff['X']`. The domain's own
    rules give the rest:

    - `offset(base, at, *steps)`: the cell of the grid `base` at the coordinates `at` shifted by the steps, or OUTSIDE;
    - `moves(player)`: for each action `player` can take here, its outcomes as (view, probability); none where the
      game gives the player no action;
    - `mobility(player)`: how many actions that is;
    - `changed(player, base, at)`: how many of those actions change the cell of the grid `base` at `at` (the entry at
      the key `at` of a map, or, `at` being None, a scalar or a list as a whole), each outcome weighted by its
      probability, worked out once for the position and player;
    - what if: `with_value(base, at, value)`, the view with one cell set; `copied(source, target)`, with every grid's
      value at `source` also at `target`. Everything else stays as it is;
    - `best(player, reading)` and `worst(player, reading)`: the highest and lowest reading expected after one of those
      actions, or the reading here when the player has none;
    - `count(player, reading)`: how many of those actions the reading is expected to hold after.

    A look-ahead's result is kept on the view, by the reading's code, the views it closes over, and the `me`, `other`
    and `here` it reads."""

    __slots__ = ("_mechanics", "_rbs", "_state", "_variables", "_memo", "_changes", "_after")

    def __init__(self, mechanics: "Mechanics", rbs: RuleBasedGame, state: State, after: "PositionView | None" = None) -> None:
        self._mechanics = mechanics
        self._rbs = rbs
        self._state = state
        self._variables: dict[str, object] | None = None
        self._memo: dict[tuple[object, ...], float] = {}
        self._changes: dict[str, dict[tuple[str, object], float]] = {}
        self._after = after

    @property
    def state(self) -> State:
        return self._state

    def __getattr__(self, name: str) -> object:
        if name[0] == "_":
            raise AttributeError(name)
        variables = self._variables
        if variables is None:
            variables = self._namespace()
        try:
            return variables[name]
        except KeyError:
            raise AttributeError(f"The position has no model {name!r}") from None

    def __getitem__(self, name: str) -> object:
        """A model read by name rather than as an attribute, for a name that is not a Python identifier.

        A game names its models to be read by people — "black may castle king side", "times this position has come
        up" — and those names are readings, offered to whatever composes expressions. An expression is Python, so a
        name with spaces in it cannot be an attribute there, and subscripting is how it is reached instead."""
        try:
            return getattr(self, name) if name.isidentifier() else (self._variables or self._namespace())[name]
        except KeyError:
            raise KeyError(f"The position has no model {name!r}") from None

    def _namespace(self) -> dict[str, object]:
        """The position's models as rules read them, worked out once; a view a move led to builds them from the position
        it came from."""
        variables = self._variables
        if variables is None:
            came_from = self._after
            if came_from is None:
                variables = self._mechanics.variables(self._state)
            else:
                variables = self._mechanics.variables_after(came_from.state, came_from._namespace(), self._state)
            self._variables = variables
        return variables

    def offset(self, base: str, at: object, *steps: int) -> object:
        """The cell of the grid `base` at the coordinates `at` shifted by the steps; OUTSIDE off the grid, or when `base`
        isn't a grid of as many dimensions as there are steps."""
        grid = getattr(self, base)
        coordinates = at if isinstance(at, tuple) else (at,)
        if not isinstance(grid, Grid) or len(coordinates) != len(steps) or not all(type(part) is int for part in coordinates):
            return OUTSIDE
        shifted = tuple(part + step for part, step in zip(coordinates, steps, strict=True))
        return grid.at(shifted) if grid.inside(shifted) else OUTSIDE

    def moves(self, player: str) -> Moves:
        return self._mechanics.moves(self._rbs, self._state, player)

    def mobility(self, player: str) -> int:
        return len(self.moves(player))

    def changed(self, player: str, base: str, at: object) -> float:
        changes = self._changes.get(player)
        if changes is None:
            changes = self._changes[player] = self._mechanics.changes(self._rbs, self._state, player)
        return changes.get((base, at), 0.0)

    def with_value(self, base: str, at: object, value: Value) -> "PositionView":
        return self._mechanics.view(self._rbs, self._mechanics.with_value(self._state, base, at, value))

    def copied(self, source: object, target: object) -> "PositionView":
        return self._mechanics.view(self._rbs, self._mechanics.copied(self._state, source, target))

    def best(self, player: str, reading: Reading) -> float:
        return self._look_ahead(BEST, player, reading)

    def worst(self, player: str, reading: Reading) -> float:
        return self._look_ahead(WORST, player, reading)

    def count(self, player: str, reading: Reading) -> float:
        return self._look_ahead(COUNT, player, reading)

    def best_changing(self, player: str, base: str, at: object, reading: Reading) -> float:
        """The most that reading comes to after one of the player's moves that changes the base at that index.

        **A look-ahead over one square rather than over everything.** `best` reads the position after every
        move a player has, which is a minimax and costs what one costs. What a tactic asks is narrower and far
        cheaper: of the moves that touch *this* square, what is the best that follows. That is the shape a
        static exchange has — take here, they take back here, take again — and nesting this alternates the
        sides of it without any of it knowing what a capture is.

        **Nothing where the player has no such move, which is not nought.** `best` falls back on the position
        in hand, because a player with no moves at all is a position that stands; that is wrong here, where
        the question was about a move that does not exist. Nought is wrong too, and measured: asked for how
        many pieces are left after an exchange, a square with no recapture answered nought and read as though
        both sides had been wiped off the board.

        So it reads as nothing at all, which is what this project means by a term that did not fire — and
        not firing is not the same as being wrong. A comparison against it is false, a sum through it is
        nothing, and the fit holds none of it against the term."""
        return self._changing(BEST, player, base, at, reading)

    def worst_changing(self, player: str, base: str, at: object, reading: Reading) -> float:
        """The least that reading comes to after one of the player's moves that changes the base at that index.

        The other half of an exchange: `best` is what the mover picks, `worst` is what they are held to."""
        return self._changing(WORST, player, base, at, reading)

    def _changing(self, kind: str, player: str, base: str, at: object, reading: Reading) -> float:
        expected = [
            math.fsum(probability * float(reading(view)) for view, probability in outcomes)  # type: ignore[arg-type]
            for outcomes in self.moves(player)
            if any((base, at) in self._mechanics.changed_parts(self._state, view.state) for view, _ in outcomes)
        ]
        if not expected:
            return math.nan
        return max(expected) if kind == BEST else min(expected)

    def _look_ahead(self, kind: str, player: str, reading: Reading) -> float:
        key = self._key(kind, player, reading)
        if key is not None and (kept := self._memo.get(key)) is not None:
            return kept
        moves = self.moves(player)
        if kind == COUNT:
            result = math.fsum(
                math.fsum(probability for view, probability in outcomes if reading(view)) for outcomes in moves
            )
        elif not moves:
            result = float(reading(self))  # type: ignore[arg-type]
        else:
            expected = [
                math.fsum(probability * float(reading(view)) for view, probability in outcomes)  # type: ignore[arg-type]
                for outcomes in moves
            ]
            result = max(expected) if kind == BEST else min(expected)
        if key is not None:
            self._memo[key] = result
        return result

    def _key(self, kind: str, player: str, reading: Reading) -> tuple[object, ...] | None:
        code = getattr(reading, "__code__", None)
        if code is None:
            return None
        closure = tuple(cell.cell_contents for cell in reading.__closure__ or ())  # type: ignore[attr-defined]
        names = reading.__globals__  # type: ignore[attr-defined]
        key = (kind, player, code, closure, names.get(ME), names.get(OTHER), names.get(HERE))
        try:
            hash(key)
        except TypeError:
            return None
        return key
