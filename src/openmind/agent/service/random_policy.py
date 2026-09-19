import random

from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.timing.model.clock import Clock
from openmind.world.model.action import Action
from openmind.world.model.state import State


class RandomPolicy:
    """Chooses uniformly among the legal actions; where players act at once, among the given player's. A clock changes
    nothing: it chooses at once."""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose(
        self, rbs: RuleBasedGame, state: State, player: str | None = None, clock: Clock | None = None, steps_played: int = 0
    ) -> Action:
        actions = rbs.actions(state, player=player)
        if not actions:
            raise ValueError("No legal action to choose from")
        return self._rng.choice(actions)
