from typing import Protocol

from openmind.structure.model.value import Value


class ErrorModel(Protocol):
    """How likely a mechanism is to say `said` when the truth is `true`, given its accuracy and how many values it could
    have said."""

    def likelihood(self, said: Value, true: Value, accuracy: float, candidates: int) -> float: ...
