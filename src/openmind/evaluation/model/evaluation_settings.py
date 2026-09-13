from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Games per baseline series, the agent's iterations, sampled positions (0 skips agreement, None takes every
    position), iteration budgets for agreement, seed."""

    games: int
    iterations: int
    positions: int | None
    budgets: tuple[int, ...]
    seed: int
