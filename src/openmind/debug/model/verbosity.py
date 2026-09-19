import logging
from dataclasses import dataclass, field

from openmind.debug.model.target import Target


@dataclass(frozen=True, slots=True)
class Verbosity:
    """Log lines at that level or above, where the target matches."""

    level: int = logging.INFO
    target: Target = field(default_factory=Target)
