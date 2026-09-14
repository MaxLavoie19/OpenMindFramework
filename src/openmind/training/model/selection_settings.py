from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SelectionSettings:
    """The positions rule sets are compared on (None takes every exact-search position), where exact search can't reach
    the iterations of the unguided searches whose values stand in for perfect play (None uses exact search), the guided
    searches' iterations and whether their rollouts follow the ratings, the margin on the rise in mean regret a removal
    may cause, the confidence of the one-sided bound on that rise and its bootstrap resamples, the seed, and the hours
    the removal passes may take (None: no limit)."""

    positions: int | None
    reference_iterations: int | None
    iterations: int
    guided_rollouts: bool
    margin: float
    confidence: float
    resamples: int
    seed: int
    max_hours: float | None
