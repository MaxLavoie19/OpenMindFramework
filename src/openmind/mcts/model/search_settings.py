from dataclasses import dataclass

from openmind.mcts.constant.mcts_constant import DEFAULT_REGRET_EXPLORATION


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """How long and how widely to search: iterations, UCT exploration weight, random seed (None is unseeded), how many
    actions a rollout plays at most before every player gets the unfinished payoff (None plays rollouts to the end), and
    the share of uniform choice mixed into regret matching where players act at once."""

    iterations: int
    exploration: float
    seed: int | None
    rollout_limit: int | None = None
    unfinished_payoff: float | None = None
    regret_exploration: float = DEFAULT_REGRET_EXPLORATION
