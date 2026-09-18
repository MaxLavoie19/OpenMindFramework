from typing import Protocol

from openmind.mcts.model.hypothesis import Hypothesis
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.model.action import Action
from openmind.world.model.state import State


class TheoryOfMind(Protocol):
    """What a player believes about what it can't be sure of, such as another player's hidden move: from the position
    it is in, hypotheses with their probabilities, summing to 1; and, where players act at once, the strategy it
    predicts another player to act will play, as (action, probability) pairs summing to 1, or None without a
    prediction.

    The beliefs are the agent's own. OMF runs agents, not the game, so nothing hides a state from a player: what a
    player doesn't know is what its theory of mind says it doesn't know."""

    def hypotheses(self, rbs: RuleBasedSystem, state: State, player: str) -> tuple[tuple[Hypothesis, float], ...]: ...

    def strategy(
        self, rbs: RuleBasedSystem, state: State, player: str, other: str
    ) -> tuple[tuple[Action, float], ...] | None: ...
