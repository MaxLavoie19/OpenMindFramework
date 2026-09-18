from dataclasses import dataclass

from openmind.mcts.constant.mcts_constant import DEFAULT_PRIOR_TEMPERATURE, DEFAULT_PUCT_EXPLORATION, UCB1, UNIFORM_PRIOR
from openmind.rbs.model.value_settings import ValueSettings
from openmind.timing.constant.timing_constant import DEFAULT_EXPECTED_STEPS, DEFAULT_TIME_RESERVE
from openmind.timing.model.time_control import TimeControl


@dataclass(frozen=True, slots=True)
class ValueDistillationSettings:
    """Self-play games to fit value rules on, held-out games to choose among the fits and measure the rules on, the
    agent's iterations, the seed, what positions are valued at (the outcome or search target), and how value rules are
    generated and fitted; and the time control self-play plays on (None plays without a clock, the iterations being the
    budget; with one, a move's budget replaces them), with the steps
    every agent's time budget estimator expects; and how every agent's search selects: `ucb1` or `puct` with its
    exploration weight, and the prior named with its temperature, an agent without the model a prior reads following the
    uniform prior."""

    games: int
    held_out_games: int
    iterations: int
    seed: int
    target: str
    values: ValueSettings
    time_control: TimeControl | None = None
    expected_steps: int = DEFAULT_EXPECTED_STEPS
    time_reserve: float = DEFAULT_TIME_RESERVE
    selection: str = UCB1
    puct_exploration: float = DEFAULT_PUCT_EXPLORATION
    prior: str = UNIFORM_PRIOR
    prior_temperature: float = DEFAULT_PRIOR_TEMPERATURE
