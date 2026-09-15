from typing import Protocol

from openmind.agent.model.domain import Domain
from openmind.mcts.model.hypothesis import Hypothesis
from openmind.world.model.action import Action
from openmind.world.model.state import State


class TheoryOfMind(Protocol):
    """What a player believes about what it can't see, such as the other players' hidden moves: from what the player
    sees, hypotheses with their probabilities, summing to 1; and, where players act at once, the strategy it predicts
    another player to act will play, as (action, probability) pairs summing to 1, or None without a prediction."""

    def hypotheses(self, domain: Domain, observed: State, player: str) -> tuple[tuple[Hypothesis, float], ...]: ...

    def strategy(
        self, domain: Domain, state: State, player: str, other: str
    ) -> tuple[tuple[Action, float], ...] | None: ...
