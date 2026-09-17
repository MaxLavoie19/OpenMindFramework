from dataclasses import dataclass

from openmind.mcts.constant.mcts_constant import DEFAULT_PRIOR_TEMPERATURE, DEFAULT_PUCT_EXPLORATION, UCB1, UNIFORM_PRIOR
from openmind.timing.constant.timing_constant import DEFAULT_EXPECTED_STEPS
from openmind.timing.model.time_control import TimeControl


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Games per baseline series, the agent's iterations, sampled positions (0 skips agreement, None takes every
    position), iteration budgets for agreement, seed, where exact search can't reach, the iterations of the unguided
    searches that stand in for perfect play (None uses exact search), whether a guided agent's rollouts follow its
    ratings, how many rollout actions a valuing agent plays before valuing a position, and, for every agent the
    evaluation builds, the most actions a rollout plays before every player gets the unfinished payoff (None plays
    rollouts to the end); and the time control the baseline series play on (None plays without a clock, the iterations
    being the budget; with one, they cap each move), with the steps every agent's time budget estimator expects; and how
    every agent's search selects: `ucb1` or `puct` with its exploration weight, and the prior named (`uniform`, `rater`
    or `value`) with its temperature, the untrained agent always following the uniform prior."""

    games: int
    iterations: int
    positions: int | None
    budgets: tuple[int, ...]
    seed: int
    reference_iterations: int | None = None
    guided_rollouts: bool = True
    rollout_actions: int = 0
    rollout_limit: int | None = None
    unfinished_payoff: float | None = None
    time_control: TimeControl | None = None
    expected_steps: int = DEFAULT_EXPECTED_STEPS
    selection: str = UCB1
    puct_exploration: float = DEFAULT_PUCT_EXPLORATION
    prior: str = UNIFORM_PRIOR
    prior_temperature: float = DEFAULT_PRIOR_TEMPERATURE
