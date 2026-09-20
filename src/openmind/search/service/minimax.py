import logging
import math

from openmind.heuristic.model.node import Node
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.search.model.guidance import Guidance
from openmind.search.model.search_settings import SearchSettings
from openmind.search.model.strategy import MoveDistribution, Strategy
from openmind.utility.service.utility import Utility
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class Minimax:
    """The planner that searches a small game to its end, each agent taking what is best for itself.

    It fits a game small enough to be read out: tic-tac-toe, an endgame, the last moves of a puzzle. Every agent is
    taken to play its own best, so a game of two is minimax and a game of more is max-n. A finished position is worth
    what the game paid; an unfinished one at the depth given is worth what the position value heuristic says, and
    nothing where there is none.

    It gives a strategy: the best move in every state it worked out, each at probability 1, so the agent can follow it
    without searching again until the opponent leaves it."""

    def __init__(self, utility: Utility) -> None:
        self._utility = utility

    def plan(
        self,
        model: object,
        knowledge_base: KnowledgeBase,
        node: Node,
        guidance: Guidance,
        settings: SearchSettings,
    ) -> Strategy | None:
        """The strategy worked out from this node; None where the game is over here or nothing could be valued."""
        moves: dict[State, MoveDistribution] = {}
        values = self._value(knowledge_base, node, guidance, settings.depth, moves, {})
        if values is None or not moves:
            return None
        logger.info("Read %s out to its end: %d states worked out", node.game.context, len(moves))  # type: ignore[union-attr]
        return Strategy(tuple(moves.items()))

    def _value(
        self,
        knowledge_base: KnowledgeBase,
        node: Node,
        guidance: Guidance,
        depth: int | None,
        moves: dict[State, MoveDistribution],
        seen: dict[State, tuple[float, ...] | None],
    ) -> tuple[float, ...] | None:
        """What the state is worth to each player, in the players' order; None where nothing could be valued."""
        if node.state in seen:
            return seen[node.state]
        game = node.game
        players = game.players()  # type: ignore[union-attr]
        legal = game.joint_actions(node.state)  # type: ignore[union-attr]
        if not legal:
            seen[node.state] = self._paid(game, node.state, players.names)
            return seen[node.state]
        if depth is not None and depth <= 0:
            seen[node.state] = self._valued(node, guidance, players.names)
            return seen[node.state]
        best: dict[int, tuple[Action, tuple[float, ...]]] = {}
        for index, actions in legal:
            for action in actions:
                following = self._following(knowledge_base, node, index, action, guidance, depth, moves, seen)
                if following is None:
                    continue
                held = best.get(index)
                if held is None or following[index] > held[1][index]:
                    best[index] = (action, following)
        if not best:
            seen[node.state] = None
            return None
        for index, (action, _) in best.items():
            if index == players.names.index(guidance.player) or len(legal) == 1:
                moves[node.state] = ((action, 1.0),)
        values = self._together(best, len(players.names))
        seen[node.state] = values
        return values

    def _following(
        self,
        knowledge_base: KnowledgeBase,
        node: Node,
        index: int,
        action: Action,
        guidance: Guidance,
        depth: int | None,
        moves: dict[State, MoveDistribution],
        seen: dict[State, tuple[float, ...] | None],
    ) -> tuple[float, ...] | None:
        """What taking that action leads to, expected over its outcomes; None where nothing could be valued."""
        game = node.game
        players = game.players()  # type: ignore[union-attr]
        joint = JointAction(((players.names[index], action),))
        outcomes = game.joint_outcomes(node.state, joint).outcomes  # type: ignore[union-attr]
        totals = [0.0] * len(players.names)
        weighed = 0.0
        for outcome, probability in outcomes:
            values = self._value(knowledge_base, node.of(outcome), guidance, None if depth is None else depth - 1, moves, seen)
            if values is None:
                continue
            weighed += probability
            for player, value in enumerate(values):
                totals[player] += probability * value
        if weighed == 0.0:
            return None
        return tuple(total / weighed for total in totals)

    def _together(self, best: dict[int, tuple[Action, tuple[float, ...]]], players: int) -> tuple[float, ...]:
        """What the state is worth to each player: what the acting players' best gives them, averaged where several
        act at once."""
        totals = [0.0] * players
        for _, values in best.values():
            for player, value in enumerate(values):
                totals[player] += value
        return tuple(total / len(best) for total in totals)

    def _paid(self, game: object, state: State, players: tuple[str, ...]) -> tuple[float, ...] | None:
        """What a finished position paid each player: a win always carries the payoff value."""
        payoff = game.players().payoff  # type: ignore[attr-defined]
        held = state.model(payoff) if state.has(payoff) else None
        paid = []
        for player in players:
            value = held.get(player) if held is not None and hasattr(held, "get") else None
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            paid.append(float(value))
        return tuple(paid)

    def _valued(self, node: Node, guidance: Guidance, players: tuple[str, ...]) -> tuple[float, ...] | None:
        """What the position value heuristic says the state is worth, or None where the caller gave none."""
        filled = guidance.position_value
        if filled is None:
            return None
        model, valuer = filled
        values = valuer.values(model, node)
        return None if values is None or len(values) != len(players) else tuple(values)
