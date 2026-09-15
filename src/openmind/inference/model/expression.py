from dataclasses import dataclass

from openmind.inference.model.aggregate import Aggregate
from openmind.inference.model.pattern import Pattern


@dataclass(frozen=True, slots=True)
class Expression:
    """A generated expression: Python source reading its position where VIEW stands; how many conditions and operations
    it combines, which the fit prices; how many actions it looks ahead; and, for a pattern count or an aggregate, its
    structure, which grows one condition or one operation at a time."""

    template: str
    clauses: int
    plies: int
    pattern: Pattern | None = None
    aggregate: Aggregate | None = None
