from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Agreement:
    """How many of the sampled positions got an optimal choice from an agent searching with this many iterations."""

    iterations: int
    positions: int
    optimal: int
