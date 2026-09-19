from dataclasses import dataclass

from openmind.rule.model.called_rule import CalledRule


@dataclass(frozen=True, slots=True)
class ScopedConstraint:
    """A constraint on three or more parameters, checked once at most one of them still has several values."""

    rule: CalledRule
    scope: tuple[str, ...]
