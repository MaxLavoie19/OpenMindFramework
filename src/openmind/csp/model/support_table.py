from dataclasses import dataclass

from openmind.world.model.value import Value


@dataclass(frozen=True, slots=True)
class SupportTable:
    """The value pairs a constraint on two parameters allows, the first parameter's value first."""

    first: str
    second: str
    allowed: frozenset[tuple[Value, Value]]
