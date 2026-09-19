from dataclasses import dataclass

from openmind.knowledge.model.source import Source
from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Evidence:
    """A source's support for one value of a variable, and how much it carries, from 0 to 1."""

    value: Value
    strength: float
    source: Source
