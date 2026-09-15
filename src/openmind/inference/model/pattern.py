from dataclasses import dataclass

from openmind.inference.model.pattern_condition import PatternCondition


@dataclass(frozen=True, slots=True)
class Pattern:
    """Conditions on variables around an index, counted over every index of the anchor base: how many indices `at` meet
    every condition."""

    anchor: str
    conditions: tuple[PatternCondition, ...]
