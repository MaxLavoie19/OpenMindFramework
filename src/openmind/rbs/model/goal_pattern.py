from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class GoalPattern:
    """What winning moves of an action needed: the conditions that held before the move, found in this many winning
    moves."""

    action: str
    conditions: tuple[PythonRule, ...]
    moves: int
