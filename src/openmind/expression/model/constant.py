from dataclasses import dataclass

from openmind.world.model.value import Value


@dataclass(frozen=True, slots=True)
class Constant:
    """A fixed value."""

    value: Value
