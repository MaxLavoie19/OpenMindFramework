from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolveStatistics:
    """What a search did: the solutions it found, the values it tried, the dead ends it hit, and the values propagation
    removed."""

    solutions: int
    assignments: int
    dead_ends: int
    pruned_values: int
