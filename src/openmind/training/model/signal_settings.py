from dataclasses import dataclass

from openmind.training.constant.signal_constant import (
    DEFAULT_ARM_EXPLORATION,
    DEFAULT_ARMS,
    DEFAULT_GOAL_LIMIT,
    DEFAULT_SIGNAL_HORIZON,
)


@dataclass(frozen=True, slots=True)
class SignalSettings:
    """How a round learns from signals: how many signals it follows at most, those with the best records, besides winning
    and the aggregations; how many plies later a position's signals are read for its targets, 0 reading them at the
    position; how much UCB explores arms it knows less about when choosing which play each other; and how many moves
    ahead the deduced goal distance looks for a win."""

    arms: int = DEFAULT_ARMS
    horizon: int = DEFAULT_SIGNAL_HORIZON
    exploration: float = DEFAULT_ARM_EXPLORATION
    goal_limit: int = DEFAULT_GOAL_LIMIT

    def __post_init__(self) -> None:
        if self.arms < 0 or self.horizon < 0 or self.exploration < 0.0:
            raise ValueError(
                "Signal settings need arms, a horizon and an exploration of 0 or more, not "
                f"{self.arms}, {self.horizon} and {self.exploration}"
            )
        if self.goal_limit < 1:
            raise ValueError(f"Signal settings need a goal limit of 1 or more, not {self.goal_limit}")
