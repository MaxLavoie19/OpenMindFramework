from dataclasses import dataclass

from openmind.expression.model.expression import Expression


@dataclass(frozen=True, slots=True)
class ScopedConstraint:
    """A constraint on three or more parameters, checked once at most one of them still has several values."""

    expression: Expression
    scope: tuple[str, ...]
