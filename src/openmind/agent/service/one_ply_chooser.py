import math
import random
from collections.abc import Sequence

from openmind.mcts.constant.mcts_constant import RANDOM_OPTION
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.position_valuer import PositionValuer
from openmind.mcts.model.search_result import SearchResult
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.model.deadline import Deadline
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class OnePlyChooser:
    """What a player choosing without searching needs: its legal moves, each legal move valued once (its outcomes valued
    for the player to act and weighed by their probabilities, a finished outcome at its payoffs), which the deduction
    fallback reads, and a random legal move, for a player out of time."""

    def __init__(self, state_reader: StateReader) -> None:
        self._state_reader = state_reader

    def legal(self, rbs: RuleBasedSystem, state: State) -> tuple[Action, ...]:
        return rbs.actions(state)

    def values(
        self,
        rbs: RuleBasedSystem,
        state: State,
        actions: Sequence[Action],
        valuer: PositionValuer,
        deadline: Deadline | None = None,
    ) -> list[float] | None:
        """Each action's value for the player to act, in the actions' order; None when the valuer can't value an outcome,
        or the deadline passed before every action was valued."""
        player = self._state_reader.player_to_act(state, rbs.players())
        values: list[float] = []
        for action in actions:
            if deadline is not None and deadline.passed():
                return None
            worth: list[float] = []
            for outcome, probability in rbs.outcomes(state, action).outcomes:
                if rbs.actions(outcome):
                    valued = valuer.values(outcome)
                    if valued is None:
                        return None
                    worth.append(probability * valued[player])
                else:
                    worth.append(probability * self._state_reader.payoffs(outcome, rbs.players())[player])
            values.append(math.fsum(worth))
        return values

    def random(self, rbs: RuleBasedSystem, state: State, rng: random.Random) -> SearchResult:
        """A random legal move, with one visit and no value: its mean payoff is NaN, nothing having been searched. No
        legal move raises ValueError."""
        actions = self.legal(rbs, state)
        if not actions:
            raise ValueError("No legal action to choose from")
        chosen = rng.choice(actions)
        player = rbs.players().names[self._state_reader.player_to_act(state, rbs.players())]
        return SearchResult(player, (ActionStatistics(chosen, 1, math.nan),), chosen, (), option=RANDOM_OPTION)
