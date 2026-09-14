from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueSettings:
    """How value rules are generated and fitted: how many single terms, the most correlated with the payoffs, are
    multiplied in pairs; how many thresholds a quantity is cut at, at most; how many own actions solo_distance() looks
    ahead (0 leaves it out); the L1 prices swept; and when a fit stops: after max_steps, or once no weight moves by more
    than tolerance."""

    pair_pool: int
    cuts: int
    solo_limit: int
    prices: tuple[float, ...]
    max_steps: int
    tolerance: float
