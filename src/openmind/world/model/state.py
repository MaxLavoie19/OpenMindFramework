from dataclasses import dataclass

from openmind.world.model.value import Value


@dataclass(frozen=True, slots=True)
class State:
    """Named variables with their values, sorted by name."""

    variables: tuple[tuple[str, Value], ...]
