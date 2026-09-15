from dataclasses import dataclass

from openmind.csp.model.variable import Variable
from openmind.rule.model.rule import Rule


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    """An action, the parameters to solve for, and the constraints every legal action satisfies: rules that give true or
    false, as Python source or as the project's own functions."""

    name: str
    variables: tuple[Variable, ...]
    constraints: tuple[Rule, ...]
