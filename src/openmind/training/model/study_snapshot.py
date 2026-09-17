from dataclasses import dataclass

from openmind.inference.model.expression import Expression
from openmind.rbs.model.value_base import ValueBase
from openmind.training.model.signal_library import SignalLibrary


@dataclass(frozen=True, slots=True)
class StudySnapshot:
    """What a game's study starts from, as it stood when the game started: the signal library, the seeds proofs have
    induced so far, and the value base pondering measures misses against (None: the mean target)."""

    library: SignalLibrary
    seeds: tuple[Expression, ...]
    reference: ValueBase | None
