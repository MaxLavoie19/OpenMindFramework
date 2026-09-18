from dataclasses import dataclass

from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.mcts.constant.mcts_constant import DEFAULT_PRIOR_TEMPERATURE, DEFAULT_PUCT_EXPLORATION, UCB1, UNIFORM_PRIOR
from openmind.timing.constant.timing_constant import DEFAULT_EXPECTED_STEPS, DEFAULT_TIME_RESERVE
from openmind.timing.model.time_control import TimeControl


@dataclass(frozen=True, slots=True)
class ContinuousTrainingSettings:
    """How continuous training runs: how many games (None: until stopped); the agents' iterations; the seed; how much
    UCB explores arms it knows less about when choosing which play each other; the rollout actions played before a
    position is valued, the rollout limit and unfinished payoff; the deduction agents fall back on (None: never), which
    walking back needs; the positions of each decisive game walked back from its end at most; the time control games
    play on, the steps an agent's estimator expects and the share of the base time it keeps in reserve; and how every
    agent's search selects."""

    games: int | None
    iterations: int
    seed: int
    arm_exploration: float
    rollout_actions: int
    rollout_limit: int | None
    unfinished_payoff: float | None
    deduction: DeductionBudget | None = None
    ponder_endings: int = 0
    time_control: TimeControl | None = None
    expected_steps: int = DEFAULT_EXPECTED_STEPS
    time_reserve: float = DEFAULT_TIME_RESERVE
    selection: str = UCB1
    puct_exploration: float = DEFAULT_PUCT_EXPLORATION
    prior: str = UNIFORM_PRIOR
    prior_temperature: float = DEFAULT_PRIOR_TEMPERATURE

    def __post_init__(self) -> None:
        if self.games is not None and self.games < 0:
            raise ValueError(f"Continuous training needs 0 games or more, not {self.games}")
        if self.arm_exploration < 0.0:
            raise ValueError(f"An arm exploration can't be negative, not {self.arm_exploration}")
        if self.ponder_endings < 0:
            raise ValueError("Walked back positions can't be negative")
        if self.ponder_endings and self.deduction is None:
            raise ValueError("Walking back needs a deduction budget")
