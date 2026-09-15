from dataclasses import dataclass
from datetime import datetime

from openmind.training.model.training_round import TrainingRound
from openmind.training.model.value_training_settings import ValueTrainingSettings


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """A value training loop so far: the domain, when it started, its settings, the rounds done, and whether every round
    is done."""

    domain: str
    created_at: datetime
    settings: ValueTrainingSettings
    rounds: tuple[TrainingRound, ...]
    complete: bool
