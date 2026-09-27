import logging
from collections.abc import Callable, Mapping, Sequence

from openmind.inference.model.example import Example
from openmind.statement.model.consequence import Consequence
from openmind.statement.model.clause import Clause
from openmind.statement.model.drawn import ALWAYS, ASKED, COLUMN, Drawn, MORE, OTHER, PLACE, ROW, STANDING, STEPPED, drawing, part
from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.grid import Grid
from openmind.structure.model.record import Record
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.statement.model.change import Change, Moved, Placed, Removed, Told
from openmind.world.model.state import State

logger = logging.getLogger(__name__)

#: What a part comes to where it cannot be drawn at all, told apart from its coming to nothing.
UNDRAWN = object()

MADE = {"Placed": Placed, "Removed": Removed, "Moved": Moved, "Told": Told}


class ConsequenceDrawer:
    """Turns what an action is said to do into what it does here.

    A consequence names its parts by where they come from rather than by what they are, so drawing them against a
    position and an action is what makes a rule about every game into a change on this board."""

    def changes(
        self,
        doing: Sequence[Consequence],
        state: State,
        action: Action,
        acting: Value,
        players: Sequence[Value],
        case: Example | None = None,
        holds: Callable[[Sequence[Clause], Example], bool] | None = None,
    ) -> tuple[Change, ...]:
        """Everything those consequences do here, in the order the game did them, where their conditions hold.

        **Made in the order the game made them**, which is not presentation: what stands somewhere is removed and
        *then* something moves onto it, which is a capture; the other way round the thing that moved is what gets
        deleted.

        **And only where their conditions hold, which nothing asked until now.** A consequence says when it
        happens — learned by the machinery that learns what a game refuses — and drawing one regardless says
        that every move takes something, clears every castling right, and moves a castling's rook. It went
        unnoticed because nothing drew consequences at all until the predictor and the constraints were joined.

        **It is handed a way to ask rather than something to hold.** Answering a condition means putting a clause
        to a case, which is the refusal learner's work — and the learner holds the hypothetical, which holds
        this, so holding a learner here would close a ring. The same shape as `_could_take`, which is handed a
        `refuses` and keeps nothing.

        Asked of a case and not of a position and an action, because building that case is a position read.
        Per consequence that would be ruinous; the caller has one already, and where it has many it can share
        one position's readings across all of its moves.

        Given no way to ask, everything draws, which is what it did before. That is not a fallback kept for
        tidiness: a caller with no learner — the predictor's own tests, a game with no constraints yet — has no
        means to answer and should not be stopped from drawing."""
        found = []
        for one in sorted(doing, key=lambda held: held.order):
            if one.when and holds is not None and case is not None and not holds(one.when, case):
                continue
            drawn = self.drawn(one, state, action, acting, players)
            if drawn is not None:
                found.append(drawn)
        return tuple(found)

    def drawn(
        self, consequence: Consequence, state: State, action: Action, acting: Value, players: Sequence[Value]
    ) -> Change | None:
        """The change that consequence makes here, or None where it names a square that isn't on the board.

        One consequence and its conditions unasked. `changes` is what a caller wants, and this is what it is
        built from."""
        parameters = dict(action.parameters)
        made = MADE[consequence.change]
        if made is Told:
            value = self._part(consequence.value, state, parameters, acting, players)
            return None if value is UNDRAWN else Told(consequence.model, value)
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
        value = self._part(consequence.value, state, parameters, acting, players)
        return None if value is UNDRAWN else Placed(consequence.model, at, value)

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
        """What that part comes to, here, or `UNDRAWN` where it cannot be drawn at all.

        **Nothing and not-drawable are different answers and were the same one.** An emptied square holds
        nothing and a promotion nobody asked for is nothing, so None is a value a drawing legitimately comes
        to. It was also what came back when a drawing could not be made — the player not acting, with nobody
        known to be acting; a count of something that is not a number — and the two were indistinguishable.
        Drawn as a value, a turn that could not be worked out was *set to nothing*, which is a position with
        no player to move and no error anywhere. Said apart, the change is simply not made, and a game whose
        turn passing could not be drawn does not pass its turn, which is a failure somebody can see."""
        if drawn is None:
            return None
        which = drawing(drawn)
        if which == ALWAYS:
            return part(drawn, 0)
        if which == ASKED:
            return parameters.get(str(part(drawn, 0)))
        if which == OTHER:
            others = [player for player in players if player != acting]
            return others[0] if len(others) == 1 else UNDRAWN
        if which == MORE:
            held = state.value(str(part(drawn, 0)))
            by = part(drawn, 1)
            return held + by if self._number(held) and self._number(by) else UNDRAWN  # type: ignore[operator]
        if which == STANDING:
            model = str(part(drawn, 0))
            at = self._at(state, model, parameters.get(str(part(drawn, 1))))
            grid = state.model(model)
            if at is None or not isinstance(grid, Grid) or not grid.inside(at):
                return UNDRAWN
            return grid.at(at)
        if which == PLACE:
            return self._held(parameters.get(str(part(drawn, 0))), str(part(drawn, 1)))
        if which == STEPPED:
            return self._stepped(
                self._held(parameters.get(str(part(drawn, 0))), str(part(drawn, 1))),
                parameters.get(str(part(drawn, 2))),
            )
        if which not in (ROW, COLUMN):
            return UNDRAWN
        at = self._at(state, self._any_grid(state), parameters.get(str(part(drawn, 0))))
        if at is None:
            return None
        return at[0] if which == ROW else at[1]

    def _held(self, value: Value, place: str) -> Value:
        """That named place of what a parameter holds.

        **The half of `Place` and `Stepped` that was never written.** The predictor learns its drawings in these
        two — `ConsequenceLearner._ways` offers nothing else for a place — and this is where a drawing becomes a
        square. Without them every grid change drew None, so a move drew no move: `after` handed back a position
        with the turn changed, the clock counted and every piece exactly where it started. Nothing raised, and
        the two halves of the design could not meet.

        A parameter of several places is a record and is asked by name. A parameter of one place *is* that place
        however the game names it, so its value is the answer."""
        return dict(value.parts).get(place) if isinstance(value, Record) else value

    def _stepped(self, held: Value, by: Value) -> Value:
        """That place moved on by what another parameter says, where both are numbers.

        A parameter holding several places cannot say by how much on its own — which of them steps is not in the
        drawing — so it says nothing rather than choosing one."""
        if isinstance(by, Record) or not self._number(held) or not self._number(by):
            return None
        return held + by  # type: ignore[operator]

    def _number(self, value: Value) -> bool:
        return isinstance(value, int | float) and not isinstance(value, bool)

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
