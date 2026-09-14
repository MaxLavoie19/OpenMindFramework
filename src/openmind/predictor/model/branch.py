from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class Branch:
    """One possible outcome of an action: its probability and its effects, a Python script whose assignments to state
    variables make the new state."""

    probability: float
    effects: PythonRule
