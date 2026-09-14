from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class ValueRule:
    """A term of a position's value and its weight: the term is a Python value rule giving a number, a boolean counting as
    0 or 1, for the player the position is valued for."""

    term: PythonRule
    weight: float
