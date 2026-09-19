import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from openmind.inference.constant.inference_constant import BEST, COUNT, HERE, ME, OTHER, OUTSIDE, WORST
from openmind.rbs.service.rule_based_system import RuleBasedSystem
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
    - `moves(player)`: for each action `player` could take if it were their turn, its outcomes as (view, probability);
    - `mobility(player)`: how many actions that is;
    - `changed(player, base, at)`: how many of those actions change the cell of the grid `base` at `at` (the entry at
      the key `at` of a map, or, `at` being None, a scalar or a list as a whole), each outcome weighted by its
      probability, worked out once for the position and player;
    - what if: `with_value(base, at, value)`, the view with one cell set; `cleared(at)`, with every grid's cell at
      `at` set to the grid's empty value, as the domain declares it; `copied(source, target)`, with every grid's value
      at `source` also at `target`; `alone(at)`, with every grid emptied but at `at`. Everything else, the player to
      act included, stays as it is;
    - `best(player, reading)` and `worst(player, reading)`: the highest and lowest reading expected after one of those
      actions, or the reading here when the player has none;
    - `count(player, reading)`: how many of those actions the reading is expected to hold after.

    A look-ahead's result is kept on the view, by the reading's code, the views it closes over, and the `me`, `other`
    and `here` it reads."""

    __slots__ = ("_mechanics", "_rbs", "_state", "_variables", "_memo", "_changes", "_after")

    def __init__(self, mechanics: "Mechanics", rbs: RuleBasedSystem, state: State, after: "PositionView | None" = None) -> None:
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

    def cleared(self, at: object) -> "PositionView":
        return self._mechanics.view(self._rbs, self._mechanics.cleared(self._rbs, self._state, at))

    def copied(self, source: object, target: object) -> "PositionView":
        return self._mechanics.view(self._rbs, self._mechanics.copied(self._state, source, target))

    def alone(self, at: object) -> "PositionView":
        return self._mechanics.view(self._rbs, self._mechanics.alone(self._rbs, self._state, at))

    def best(self, player: str, reading: Reading) -> float:
        return self._look_ahead(BEST, player, reading)

    def worst(self, player: str, reading: Reading) -> float:
        return self._look_ahead(WORST, player, reading)

    def count(self, player: str, reading: Reading) -> float:
        return self._look_ahead(COUNT, player, reading)

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
