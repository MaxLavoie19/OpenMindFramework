from dataclasses import dataclass

from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.model.pondering_settings import PonderingSettings


@dataclass(frozen=True, slots=True)
class ValueDistillationSettings:
    """Self-play games to fit value rules on, held-out games to choose among the fits and measure the rules on, the
    agent's iterations, the seed, what positions are valued at (the outcome or search target), how value rules are
    generated and fitted, and how many positions are pondered before fitting and within what budget (None ponders
    none)."""

    games: int
    held_out_games: int
    iterations: int
    seed: int
    target: str
    values: ValueSettings
    pondering: PonderingSettings | None = None
