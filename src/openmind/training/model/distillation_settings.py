from dataclasses import dataclass

from openmind.rbs.model.induction_settings import InductionSettings


@dataclass(frozen=True, slots=True)
class DistillationSettings:
    """Self-play games to learn from, held-out games to measure on, the agent's iterations, the seed, and induction."""

    games: int
    held_out_games: int
    iterations: int
    seed: int
    induction: InductionSettings
