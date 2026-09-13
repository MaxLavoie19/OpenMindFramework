from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Agreement:
    """How many of the sampled positions got an optimal choice from an agent searching with this many iterations, and
    the mean time each choice took."""

    iterations: int
    positions: int
    optimal: int
    seconds_per_choice: float
