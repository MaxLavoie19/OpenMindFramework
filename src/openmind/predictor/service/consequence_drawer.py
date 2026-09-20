import logging
from collections.abc import Mapping, Sequence

from openmind.predictor.model.consequence import Consequence
from openmind.predictor.model.drawn import Always, Asked, Column, Drawn, More, Other, Row, Standing
from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.grid import Grid
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.change import Change, Moved, Placed, Removed, Told
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

MADE = {"Placed": Placed, "Removed": Removed, "Moved": Moved, "Told": Told}


class ConsequenceDrawer:
    """Turns what an action is said to do into what it does here.

    A consequence names its parts by where they come from rather than by what they are, so drawing them against a
    position and an action is what makes a rule about every game into a change on this board."""

    def drawn(
        self, consequence: Consequence, state: State, action: Action, acting: Value, players: Sequence[Value]
    ) -> Change | None:
        """The change that consequence makes here, or None where it names a square that isn't on the board."""
        parameters = dict(action.parameters)
        made = MADE[consequence.change]
        if made is Told:
            return Told(consequence.model, self._part(consequence.value, state, parameters, acting, players))
        at = self._cell(consequence.where, state, parameters, acting, players)
        if at is None or not self._inside(state, consequence.model, at):
            return None
        if made is Removed:
            return Removed(consequence.model, at)
        if made is Moved:
            onto = self._cell(consequence.onto, state, parameters, acting, players)
            if onto is None or not self._inside(state, consequence.model, onto):
                return None
            return Moved(consequence.model, at, onto)
        return Placed(consequence.model, at, self._part(consequence.value, state, parameters, acting, players))

    def _cell(
        self,
        where: Sequence[Drawn],
        state: State,
        parameters: Mapping[str, Value],
        acting: Value,
        players: Sequence[Value],
    ) -> Coordinates | None:
        drawn = tuple(self._part(one, state, parameters, acting, players) for one in where)
        if not drawn or any(not isinstance(one, int) or isinstance(one, bool) for one in drawn):
            return None
        return tuple(int(one) for one in drawn)  # type: ignore[arg-type]

    def _part(
        self,
        drawn: Drawn | None,
        state: State,
        parameters: Mapping[str, Value],
        acting: Value,
        players: Sequence[Value],
    ) -> Value:
        """What that part comes to, here."""
        if drawn is None:
            return None
        if isinstance(drawn, Always):
            return drawn.value
        if isinstance(drawn, Asked):
            return parameters.get(drawn.parameter)
        if isinstance(drawn, Other):
            others = [player for player in players if player != acting]
            return others[0] if len(others) == 1 else None
        if isinstance(drawn, More):
            held = state.value(drawn.model)
            return held + drawn.by if isinstance(held, int | float) and not isinstance(held, bool) else None
        if isinstance(drawn, Standing):
            at = self._at(state, drawn.model, parameters.get(drawn.parameter))
            grid = state.model(drawn.model)
            if at is None or not isinstance(grid, Grid) or not grid.inside(at):
                return None
            return grid.at(at)
        at = self._at(state, self._any_grid(state), parameters.get(drawn.parameter))
        if at is None:
            return None
        return at[0] if isinstance(drawn, Row) else at[1]

    def _at(self, state: State, model: str | None, value: Value) -> Coordinates | None:
        """The cell a parameter points at, where it names one."""
        if model is None or not isinstance(value, str):
            return None
        grid = state.model(model)
        if not isinstance(grid, Grid) or grid.aliases is None:
            return None
        try:
            return grid.aliases.to_coordinates(value)
        except (ValueError, KeyError):
            return None

    def _any_grid(self, state: State) -> str | None:
        return next((name for name, model in state.models if isinstance(model, Grid)), None)

    def _inside(self, state: State, model: str, at: Coordinates) -> bool:
        grid = state.model(model)
        return isinstance(grid, Grid) and grid.inside(at)
