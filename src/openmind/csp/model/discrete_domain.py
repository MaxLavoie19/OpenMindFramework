from dataclasses import dataclass

from openmind.world.model.value import Value


@dataclass(frozen=True, slots=True)
class DiscreteDomain:
    """The finite values a variable can take."""

    values: tuple[Value, ...]
