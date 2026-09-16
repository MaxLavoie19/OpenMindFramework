from typing import Protocol


class TimeSource(Protocol):
    """Where time comes from: seconds that never run backwards. Only the difference between two readings means anything,
    not a reading on its own."""

    def now(self) -> float: ...
