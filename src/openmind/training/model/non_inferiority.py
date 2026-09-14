from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NonInferiority:
    """A paired comparison of regret over positions: the mean of one set's regret minus the other's, the one-sided upper
    bound of that mean, the margin, and whether the bound is below the margin, that is, whether the first set plays no
    worse than the second by more than the margin."""

    positions: int
    regret_difference: float
    upper_bound: float
    margin: float
    holds: bool
