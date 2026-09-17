import math
import random

from openmind.agent.model.domain import Domain
from openmind.agent.model.game_summary import GameSummary
from openmind.predictor.service.predictor import Predictor
from openmind.training.model.played_game import PlayedGame
from openmind.world.model.state import State


class GameReplayer:
    """Plays a remembered game again from its actions and its outcome seed, drawing each outcome as the game drew it, so
    its positions come back exactly as they were. Its search values aren't remembered: each position gets the mean of
    the final payoffs instead."""

    def __init__(self, predictor: Predictor) -> None:
        self._predictor = predictor

    def replay(self, domain: Domain, summary: GameSummary) -> PlayedGame:
        """The game with its positions, payoffs, arms and actions. A summary without an outcome seed raises ValueError."""
        if len(summary.seeds) < 2:
            raise ValueError(f"{summary.label} has no outcome seed to replay it from")
        rng = random.Random(summary.seeds[1])
        state = domain.initial_state
        states: list[State] = []
        for action in summary.actions:
            states.append(state)
            outcomes = self._predictor.predict(domain.transitions, state, action).outcomes
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
        mean = math.fsum(summary.payoffs) / len(summary.payoffs)
        return PlayedGame(
            (),
            tuple(states),
            (mean,) * len(states),
            summary.payoffs,
            tuple(model.name for model in summary.models),
            summary.actions,
            summary.time_control,
            summary.seconds,
            summary.budgets,
            summary.clocks,
            summary.flagged,
            summary.seeds[0],
            summary.seeds[1],
            summary.ending,
        )
