from dataclasses import dataclass

from openmind.rule.model.compiled_rule import CompiledRule


@dataclass(frozen=True, slots=True)
class ScopedConstraint:
    """A constraint on three or more parameters, checked once at most one of them still has several values."""

    rule: CompiledRule
    scope: tuple[str, ...]
