import logging

from openmind.heuristic.model.node import Node
from openmind.heuristic.model.position_valuer import PositionValuer
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class SuccessorMoveRater:
    """What each action is worth, from what the positions it leads to are worth.

    **This is how a move value exists before a single move term has been fitted.** Rating moves and valuing
    positions are two tasks with two rulesets, and the move one is the harder to learn: a position term reads
    the position, a move term has to read the action, which is machinery that does not exist yet. Meanwhile
    everything that wants a policy wants a move value — the tree search's priors, the greedy planner's first
    question — so the move task stayed empty and they went without.

    A position value answers the move question at the cost of one step: take the action, weigh what it may
    lead to by how likely each is, and that is what the action is worth. Nothing new is learned and nothing is
    guessed. It is a *derivation*, and it should be said as one rather than registered as a move-value model
    somebody might mistake for something fitted.

    **A finished position is worth what the game paid there**, read from the game's own payoff, never a
    model's opinion of it. An unfinished one is worth what the position value says, and where that says
    nothing, so does this: None is knows nothing, not worth zero.

    **Where more than one player acts at once it rates nothing.** What an action leads to is then not settled
    by that action alone, and there is no honest one-step answer — the same limit the deducer already states.
    Declining is the answer; pretending the others pass is not."""

    def __init__(self, position_valuer: PositionValuer[object] | None = None) -> None:
        # The valuer this fills the move task with, where it is filling it. A caller that carries its own —
        # a planner handed one with its guidance — asks `through` instead and gives this none.
        self._valuer = position_valuer

    def rate(
        self, model: object, node: Node, actions: tuple[Action, ...], player: str
    ) -> tuple[float | None, ...]:
        """What each action is worth to that player, in the actions' order, through the valuer this was built
        with — which is the move value task, filled by the position value ruleset it is given.

        Built with no valuer it knows nothing, which is what it should say rather than pretending to rate."""
        if self._valuer is None:
            return (None,) * len(actions)
        return self.through(self._valuer, model, node, actions, player)

    def through(
        self,
        valuer: PositionValuer[object],
        model: object,
        node: Node,
        actions: tuple[Action, ...],
        player: str,
    ) -> tuple[float | None, ...]:
        """The same, through a valuer given here rather than the one this was built with.

        A planner is handed its position value with the guidance and it is not this one's to assume, so the
        walk is offered both ways rather than written twice."""
        game = node.game
        if game is None:
            return (None,) * len(actions)
        players = game.players()  # type: ignore[union-attr]
        names = players.names
        if player not in names:
            return (None,) * len(actions)
        at = names.index(player)
        if len(game.joint_actions(node.state)) > 1:  # type: ignore[union-attr]
            logger.debug("Rating no move for %s: somebody else is choosing at the same moment", player)
            return (None,) * len(actions)
        return tuple(self._expected(valuer, model, node, at, one, names) for one in actions)

    def _expected(
        self,
        valuer: PositionValuer[object],
        model: object,
        node: Node,
        at: int,
        action: Action,
        names: tuple[str, ...],
    ) -> float | None:
        """What taking that action is worth to that player, weighed over the outcomes it may have."""
        game = node.game
        joint = JointAction(((names[at], action),))
        outcomes = game.joint_outcomes(node.state, joint).outcomes  # type: ignore[union-attr]
        total, weighed = 0.0, 0.0
        for outcome, probability in outcomes:
            values = self._worth(valuer, model, node.of(outcome), names)
            if values is None:
                continue
            weighed += probability
            total += probability * values[at]
        return total / weighed if weighed else None

    def _worth(
        self, valuer: PositionValuer[object], model: object, node: Node, names: tuple[str, ...]
    ) -> tuple[float, ...] | None:
        """What that position is worth to each player: what the game paid where it is finished, and what the
        position value says where it is not."""
        game = node.game
        if not game.joint_actions(node.state):  # type: ignore[union-attr]
            return self._paid(game, node.state, names)
        values = valuer.values(model, node)
        return None if values is None or len(values) != len(names) else tuple(values)

    def _paid(self, game: object, state: State, names: tuple[str, ...]) -> tuple[float, ...] | None:
        """What a finished position paid each player, read from the game's own payoff and never guessed."""
        payoff = game.players().payoff  # type: ignore[attr-defined]
        held = state.model(payoff) if state.has(payoff) else None
        paid = []
        for player in names:
            value = held.get(player) if held is not None and hasattr(held, "get") else None
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            paid.append(float(value))
        return tuple(paid)
