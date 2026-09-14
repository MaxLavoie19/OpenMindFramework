from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class StateDomain:
    """The values a variable can take in a state: a value rule that reads the state and the problem's definitions and
    gives them, in order; a value given twice counts once."""

    rule: PythonRule
