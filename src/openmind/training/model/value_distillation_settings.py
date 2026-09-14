from dataclasses import dataclass

from openmind.rbs.model.value_settings import ValueSettings


@dataclass(frozen=True, slots=True)
class ValueDistillationSettings:
    """Self-play games to fit value rules on, held-out games to choose among the fits and measure the rules on, the
    agent's iterations, the seed, what positions are valued at (the outcome or search target), and how value rules are
    generated and fitted."""

    games: int
    held_out_games: int
    iterations: int
    seed: int
    target: str
    values: ValueSettings
