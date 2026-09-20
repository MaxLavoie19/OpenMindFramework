import logging
import random

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy
from openmind.world.model.action import Action

logger = logging.getLogger(__name__)


class RandomPicker:
    """The optimizer that picks one of the valid actions at random: what a policy does when there is no time to think.

    It reads no model: the game's own legal actions are its candidates, so what it gives is legal by construction. Its
    random source is given once, so a run can be repeated."""

    def __init__(self, source: random.Random | None = None) -> None:
        self._source = random.Random() if source is None else source

    def optimize(self, model: object, policy: Policy, node: Node, player: str) -> Action | None:
        """One of the player's legal actions, picked at random; None where the player has none."""
        actions = node.game.actions(node.state, player=player)  # type: ignore[union-attr]
        if not actions:
            return None
        chosen = self._source.choice(actions)
        logger.debug("Picked %s at random among %d legal actions", chosen.name, len(actions))
        return chosen  # type: ignore[no-any-return]
