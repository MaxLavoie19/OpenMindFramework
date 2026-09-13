from dataclasses import dataclass

from openmind.world.model.value import Value


@dataclass(frozen=True, slots=True)
class Action:
    """The thing performed, with its parameters sorted by name."""

    name: str
    parameters: tuple[tuple[str, Value], ...]
