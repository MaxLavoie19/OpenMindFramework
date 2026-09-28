import logging
from collections.abc import Sequence

from openmind.heuristic.model.move_rater import MoveRater
from openmind.heuristic.model.node import Node
from openmind.world.model.action import Action

logger = logging.getLogger(__name__)


class CascadeMoveRater:
    """Asks each of several raters in turn and takes the first that has an opinion here.

    **A rater that has to answer everywhere is one model pretending to be a library.** What is actually known
    about a game is not evenly spread: an opening is a table of lines somebody memorised, a middlegame is
    judgement, and an ending is a procedure that applies to a particular handful of material and to nothing
    else. A single model over all three is a model of none of them, averaged. Letting a rater decline is what
    lets each be about what it is about, and it is why `MoveRater` returning None for an action has always
    meant *knows nothing* rather than *worth nothing*.

    **It declines by decision and not by action, and that is forced.** Two raters' numbers are on unrelated
    scales — one fitted on payoffs, another on how often somebody chose a move — so taking one rater's score
    for this action and another's for that one compares numbers that were never comparable, and the best
    action comes out of the arithmetic of two different languages. So the first rater with anything to say
    about *any* action here answers the whole decision.

    **The order is given, never decided here.** Which rater should go first is a question for measurement —
    held-out decisions for one that predicts a player, games for one that plays well — and those are
    different currencies that must not be mixed, so neither is built in. This takes the order it is handed
    and respects it. A cascade is itself a rater, so cascades compose and nothing downstream can tell one
    rater from a stack of them.

    Where every rater declines, so does this: nobody knowing anything is a thing to say, and saying nothing
    is how it is said."""

    def __init__(self, raters: Sequence[tuple[object, MoveRater[object]]], named: Sequence[str] = ()) -> None:
        # Each rater with the model it runs, in the order they are to be asked, and what to call them. The
        # names are for saying which one answered, which is the whole of a cascade's explanation of itself.
        self._raters = tuple(raters)
        self._named = tuple(named) if named else tuple(f"rater {at + 1}" for at in range(len(raters)))

    def rate(
        self, model: object, node: Node, actions: tuple[Action, ...], player: str
    ) -> tuple[float | None, ...]:
        """What each action is worth, from the first rater with anything to say about this decision.

        `model` is ignored: each rater in the cascade carries its own, since the point of a cascade is that
        the layers are different models and not one model consulted differently."""
        for at, (held, rater) in enumerate(self._raters):
            rated = rater.rate(held, node, actions, player)
            if any(one is not None for one in rated):
                logger.debug("%s answered for %s", self._named[at], player)
                return rated
        return (None,) * len(actions)

    def answered(
        self, node: Node, actions: tuple[Action, ...], player: str
    ) -> str:
        """Which rater answered here, or empty where none did.

        Kept apart from `rate` because a caller asking who decided is asking a different question from what
        to play, and the answer is what makes a cascade's decision readable rather than merely correct."""
        for at, (held, rater) in enumerate(self._raters):
            if any(one is not None for one in rater.rate(held, node, actions, player)):
                return self._named[at]
        return ""

    @property
    def layers(self) -> tuple[str, ...]:
        """The raters that will be consulted, in order — the cascade's own account of itself."""
        return self._named
