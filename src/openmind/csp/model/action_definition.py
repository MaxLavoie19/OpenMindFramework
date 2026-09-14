from dataclasses import dataclass

from openmind.csp.model.variable import Variable
from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    """An action, the parameters to solve for, and the constraints every legal action satisfies: Python rules that give
    true or false."""

    name: str
    variables: tuple[Variable, ...]
    constraints: tuple[PythonRule, ...]
