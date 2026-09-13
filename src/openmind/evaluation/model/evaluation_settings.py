from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Games per baseline series, the agent's iterations, sampled positions, iteration budgets for agreement, seed."""

    games: int
    iterations: int
    positions: int
    budgets: tuple[int, ...]
    seed: int
