from dataclasses import dataclass

from openmind.rbs.model.value_settings import ValueSettings
from openmind.training.model.pondering_settings import PonderingSettings
from openmind.training.model.signal_settings import SignalSettings


@dataclass(frozen=True, slots=True)
class ValueDistillationSettings:
    """Self-play games to fit value rules on, held-out games to choose among the fits and measure the rules on, the
    agent's iterations, the seed, what positions are valued at (the outcome, search or signals target), how value rules
    are generated and fitted, how many positions are pondered before fitting and within what budget (None ponders none),
    and, with the signals target, how the round follows signals (None takes the defaults)."""

    games: int
    held_out_games: int
    iterations: int
    seed: int
    target: str
    values: ValueSettings
    pondering: PonderingSettings | None = None
    signals: SignalSettings | None = None
