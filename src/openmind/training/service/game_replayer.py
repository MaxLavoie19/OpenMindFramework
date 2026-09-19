import math
import random

from openmind.agent.model.game_summary import GameSummary
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.training.model.played_game import PlayedGame
from openmind.world.model.state import State


class GameReplayer:
    """Plays a remembered game again from its actions and its outcome seed, drawing each outcome as the game drew it, so
    its positions come back exactly as they were. Its search values aren't remembered: each position gets the mean of
    the final payoffs instead."""

    def replay(self, rbs: RuleBasedGame, summary: GameSummary) -> PlayedGame:
        """The game with the positions its moves were played from, its payoffs, arms and actions. A summary without an
        outcome seed raises ValueError."""
        states = self.positions(rbs, summary)[:-1]
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

    def positions(self, rbs: RuleBasedGame, summary: GameSummary) -> tuple[State, ...]:
        """Every position of the game, from the start to the position its last move led to: one more than its moves. A
        summary without an outcome seed raises ValueError."""
        if len(summary.seeds) < 2:
            raise ValueError(f"{summary.label} has no outcome seed to replay it from")
        rng = random.Random(summary.seeds[1])
        state = rbs.start()
        states: list[State] = [state]
        for action in summary.actions:
            outcomes = rbs.outcomes(state, action).outcomes
            (state,) = rng.choices([outcome for outcome, _ in outcomes], weights=[probability for _, probability in outcomes])
            states.append(state)
        return tuple(states)
