import logging
from collections.abc import Callable, Mapping, Sequence

from openmind.inference.model.example import Example
from openmind.inference.service.candidate_readings import CandidateReadings
from openmind.predictor.service.consequence_drawer import ConsequenceDrawer
from openmind.rule.model.clause import Clause
from openmind.rule.model.consequence import Consequence
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.changer import Changer

logger = logging.getLogger(__name__)


class DrawnEffects:
    """What OMF has learned an action does, made to happen: `ConsequenceCaller`, filled.

    **A game that cannot say what its actions do cannot be played.** `EffectsRunner` raises without an effects
    rule, so everything else a learned game has — its legality, its values, its start — buys nothing until this
    exists. It was the last thing standing between what the predictor works out and a game anyone can play.

    **It needs no compiler, for the reason the clause caller gives.** Drawing a change from an action is what
    the predictor already does to predict; turning consequences into Python would be a second answer to the
    same question, and the two would drift. Here the consequences that were learned are the consequences that
    are run.

    **Who is acting is read off the position and never assumed.** A consequence can say *the player not
    acting* — whose turn it becomes — and that cannot be drawn without knowing whose turn it is now. Given no
    way to read it, those consequences draw nothing rather than drawing somebody arbitrary, and a game whose
    turn passing was learned that way simply does not pass its turn. That is a visible failure instead of a
    quiet wrong one.

    **A consequence whose conditions the position does not meet does not happen.** Without asking, every move
    takes something, clears every castling right and moves a castling's rook. Answering means putting a clause
    to a case, which is the refusal learner's work, so a way of asking is handed in rather than held — the same
    shape the drawer itself uses, and for the same reason.

    It keeps nothing: built once, it is given the consequences and the position on every call."""

    def __init__(
        self,
        consequence_drawer: ConsequenceDrawer | None = None,
        changer: Changer | None = None,
        readings: CandidateReadings | None = None,
        acting: Callable[[State], Value] | None = None,
        players: Sequence[Value] = (),
        holds: Callable[[Sequence[Clause], Example], bool] | None = None,
    ) -> None:
        self._drawer = ConsequenceDrawer() if consequence_drawer is None else consequence_drawer
        self._changer = Changer() if changer is None else changer
        self._readings = readings
        self._acting = acting
        self._players = tuple(players)
        self._holds = holds

    def after(
        self, consequences: Sequence[Consequence], state: State, parameters: Mapping[str, Value], action: str
    ) -> State:
        """The position those consequences leave, having drawn each from the action and made them in order."""
        played = Action(action, tuple(sorted(parameters.items())))
        changes = self._drawer.changes(
            consequences,
            state,
            played,
            self._whose(state),
            self._players,
            self._case(state, played),
            self._holds,
        )
        logger.debug("%s does %d of %d things it was learned to do", action, len(changes), len(consequences))
        return self._changer.applied(state, changes)

    def _whose(self, state: State) -> Value:
        """Whose action this is, read off the position; nothing where the caller gave no way to read it."""
        return None if self._acting is None else self._acting(state)

    def _case(self, state: State, action: Action) -> Example | None:
        """That candidate as the conditions would be put to it; nothing where there is nothing to read it with.

        Read once per action and shared by every consequence, because reading a position per consequence is
        what makes a whole-structure vocabulary unaffordable."""
        if self._readings is None or self._holds is None:
            return None
        return Example(self._readings.read(state, action), False, state)
