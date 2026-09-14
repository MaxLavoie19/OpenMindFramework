from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """Games per baseline series, the agent's iterations, sampled positions (0 skips agreement, None takes every
    position), iteration budgets for agreement, seed, where exact search can't reach, the iterations of the unguided
    searches that stand in for perfect play (None uses exact search), and whether a guided agent's rollouts follow its
    ratings."""

    games: int
    iterations: int
    positions: int | None
    budgets: tuple[int, ...]
    seed: int
    reference_iterations: int | None = None
    guided_rollouts: bool = True
