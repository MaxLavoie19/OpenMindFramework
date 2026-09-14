from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """How long and how widely to search: iterations, UCT exploration weight, random seed (None is unseeded), and how
    many actions a rollout plays at most before every player gets the unfinished payoff (None plays rollouts to the
    end)."""

    iterations: int
    exploration: float
    seed: int | None
    rollout_limit: int | None = None
    unfinished_payoff: float | None = None
