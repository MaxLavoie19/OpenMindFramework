from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ChoiceMeasure:
    """An agent's choice in one position: whether it was optimal, the share of the root's visits on optimal actions, its
    regret, and the seconds the search took."""

    optimal: bool
    optimal_visit_share: float
    regret: float
    seconds: float
