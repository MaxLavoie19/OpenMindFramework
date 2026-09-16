from dataclasses import dataclass

from openmind.timing.model.time_source import TimeSource


@dataclass(frozen=True, slots=True)
class Deadline:
    """A moment on a time source, carried with its source, so whatever must stop by then — a search, a plan — can be
    handed the deadline alone."""

    at: float
    source: TimeSource

    @classmethod
    def after(cls, seconds: float, source: TimeSource) -> "Deadline":
        """The moment that many seconds from now."""
        return cls(source.now() + seconds, source)

    def remaining(self) -> float:
        """The seconds left until the deadline, below 0 once it has passed."""
        return self.at - self.source.now()

    def passed(self) -> bool:
        """Whether the deadline has been reached."""
        return self.remaining() <= 0.0
