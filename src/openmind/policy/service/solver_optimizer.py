import logging

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy
from openmind.world.model.action import Action

logger = logging.getLogger(__name__)


class SolverOptimizer:
    """The optimizer that lets the constraint solver work the values out, taking the first action it settles on.

    An optimizer can be a special kind of CSP: the solver assigns the parameters one at a time under the game's
    constraints, so what it gives is legal by construction and nothing has to be checked afterwards. It reads no model
    of its own."""

    def optimize(self, model: object, policy: Policy, node: Node, player: str) -> Action | None:
        """The first action the solver settles on for the player; None where it can settle on none."""
        actions = node.game.actions(node.state, limit=1, player=player)  # type: ignore[union-attr]
        if not actions:
            return None
        logger.debug("The solver settled on %s", actions[0].name)
        return actions[0]  # type: ignore[no-any-return]
