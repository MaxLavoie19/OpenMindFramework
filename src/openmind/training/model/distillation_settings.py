from dataclasses import dataclass

from openmind.rbs.model.generation_settings import GenerationSettings


@dataclass(frozen=True, slots=True)
class DistillationSettings:
    """Self-play games to discover rules in, held-out games to validate and measure them on, the agent's iterations, the
    seed, and how rules are generated."""

    games: int
    held_out_games: int
    iterations: int
    seed: int
    generation: GenerationSettings
