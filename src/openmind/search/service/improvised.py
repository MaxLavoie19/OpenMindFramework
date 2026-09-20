import logging

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.policy.model.optimizer import Optimizer
from openmind.policy.model.policy_picker import PolicyPicker
from openmind.search.model.guidance import Guidance
from openmind.search.model.search_settings import SearchSettings
from openmind.search.model.strategy import Strategy

logger = logging.getLogger(__name__)


class Improvised:
    """The planner that doesn't plan: the policy worth following is picked, its optimizer gives the next best step, and
    that is the strategy.

    Some work is improvised rather than planned — a conversation is — and some moments leave no time to think. It is
    also what an agent falls back on where no search fits the level."""

    def __init__(self, policy_picker: PolicyPicker[object], optimizer: Optimizer[object]) -> None:
        self._picker = policy_picker
        self._optimizer = optimizer

    def plan(
        self,
        model: object,
        knowledge_base: KnowledgeBase,
        node: Node,
        guidance: Guidance,
        settings: SearchSettings,
    ) -> Strategy | None:
        """The strategy of playing the optimizer's action in this state; None where no policy has an action here."""
        policies = guidance.policies or (Policy("", node.game.context_id),)  # type: ignore[union-attr]
        for policy in self._picker.pick(model, node, policies, guidance.player):
            action = self._optimizer.optimize(model, policy, node, guidance.player)
            if action is not None:
                logger.debug(
                    "Improvised %s for %s%s", action.name, guidance.player, f" by {policy.name}" if policy.name else ""
                )
                return Strategy.of(node.state, action)
        logger.debug("Nothing to improvise for %s here", guidance.player)
        return None
