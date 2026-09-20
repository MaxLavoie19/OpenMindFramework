import logging
import math

from openmind.heuristic.model.node import Node
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.structure.model.map import Map

logger = logging.getLogger(__name__)


class Utility:
    """What a move is worth to a player: the value of each of its outcomes times that outcome's probability, summed.

    An outcome's value is its worth on every goal, weighed by the preferences held for them. A game's payoff is the
    goal every game has, so a context with no goal of its own is valued by its payoffs alone. Preferences live in the
    knowledge base, so the agent can look back on what it preferred.

    It keeps nothing: the knowledge base and the node are given with every call."""

    def of(
        self,
        knowledge_base: KnowledgeBase,
        node: Node,
        outcomes: OutcomeDistribution,
        player: str,
        payoff: str,
        holder: tuple[str, ...] = (),
        role: str = "",
    ) -> float | None:
        """What the move leading to those outcomes is worth to the player; None where no outcome could be valued."""
        valued = [
            (value, probability)
            for outcome, probability in outcomes.outcomes
            if (value := self.value(knowledge_base, node, outcome, player, payoff, holder, role)) is not None
        ]
        if not valued:
            return None
        utility = math.fsum(value * probability for value, probability in valued)
        logger.debug("The move is worth %.4g to %s over %d outcomes", utility, player, len(valued))
        return utility

    def value(
        self,
        knowledge_base: KnowledgeBase,
        node: Node,
        outcome: object,
        player: str,
        payoff: str,
        holder: tuple[str, ...] = (),
        role: str = "",
    ) -> float | None:
        """What one outcome is worth to the player: its worth on every goal, weighed by the preferences held for them.
        None where nothing it is worth could be read."""
        context = knowledge_base.context_named(node.game.context) if node.game is not None else None  # type: ignore[union-attr]
        goals = () if context is None else knowledge_base.goals(context.id)
        worth: list[float] = []
        for goal in goals:
            preference = knowledge_base.preference(goal.id, holder, role)
            if preference is None:
                continue
            reached = self._reached(outcome, goal.name, player)
            if reached is not None:
                worth.append(preference.weight * reached)
        if worth:
            return math.fsum(worth)
        return self._payoff(outcome, player, payoff)

    def _reached(self, outcome: object, goal: str, player: str) -> float | None:
        """How far the outcome reaches the goal, as the state says: a model named after the goal, a Map by player or a
        number; None where the state says nothing about it."""
        if not outcome.has(goal):  # type: ignore[attr-defined]
            return None
        held = outcome.model(goal)  # type: ignore[attr-defined]
        value = held.get(player) if isinstance(held, Map) else getattr(held, "value", None)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)

    def _payoff(self, outcome: object, player: str, payoff: str) -> float | None:
        held = outcome.model(payoff) if outcome.has(payoff) else None  # type: ignore[attr-defined]
        value = held.get(player) if isinstance(held, Map) else None
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)
