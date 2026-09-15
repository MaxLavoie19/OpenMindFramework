import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from openmind.agent.model.domain import Domain
from openmind.inference.constant.inference_constant import BEST, COUNT, HERE, ME, OTHER, OUTSIDE, WORST
from openmind.world.model.state import State

if TYPE_CHECKING:
    from openmind.inference.service.mechanics import Mechanics

type Reading = Callable[["PositionView"], object]
type Moves = tuple[tuple[tuple["PositionView", float], ...], ...]


class PositionView:
    """A position as a generated expression reads it. Its variables are attributes, gathered the way rules see them:
    `view.turn`, `view.cell[2, 3]`. The domain's own rules give the rest:

    - `offset(base, at, *steps)`: the variable of `base` at index `at` shifted by the steps, or OUTSIDE;
    - `moves(player)`: for each action `player` could take if it were their turn, its outcomes as (view, probability);
    - `mobility(player)`: how many actions that is;
    - `best(player, reading)` and `worst(player, reading)`: the highest and lowest reading expected after one of those
      actions, or the reading here when the player has none;
    - `count(player, reading)`: how many of those actions the reading is expected to hold after.

    A look-ahead's result is kept on the view, by the reading's code, the views it closes over, and the `me`, `other`
    and `here` it reads."""

    __slots__ = ("_mechanics", "_domain", "_state", "_variables", "_memo")

    def __init__(self, mechanics: "Mechanics", domain: Domain, state: State) -> None:
        self._mechanics = mechanics
        self._domain = domain
        self._state = state
        self._variables: dict[str, object] | None = None
        self._memo: dict[tuple[object, ...], float] = {}

    @property
    def state(self) -> State:
        return self._state

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._variables is None:
            self._variables = self._mechanics.variables(self._state)
        try:
            return self._variables[name]
        except KeyError:
            raise AttributeError(f"The position has no variable {name!r}") from None

    def offset(self, base: str, at: object, *steps: int) -> object:
        indices = at if isinstance(at, tuple) else (at,)
        if len(indices) != len(steps) or not all(isinstance(index, int) for index in indices):
            return OUTSIDE
        shifted = tuple(index + step for index, step in zip(indices, steps, strict=True))
        variables = getattr(self, base)
        if not isinstance(variables, dict):
            return OUTSIDE
        return variables.get(shifted if len(shifted) > 1 else shifted[0], OUTSIDE)

    def moves(self, player: str) -> Moves:
        return self._mechanics.moves(self._domain, self._state, player)

    def mobility(self, player: str) -> int:
        return len(self.moves(player))

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
