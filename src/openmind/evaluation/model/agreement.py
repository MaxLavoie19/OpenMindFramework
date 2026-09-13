from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Agreement:
    """How an agent searching with this many iterations did on the sampled positions: how many of its choices were
    optimal, the mean share of its root visits that went to optimal actions, the mean exact value its choices lost
    (regret), and the mean time each choice took."""

    iterations: int
    positions: int
    optimal: int
    optimal_visit_share: float
    mean_regret: float
    seconds_per_choice: float
