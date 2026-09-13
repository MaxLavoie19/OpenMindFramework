from dataclasses import dataclass

from openmind.csp.model.variable import Variable
from openmind.expression.model.expression import Expression


@dataclass(frozen=True, slots=True)
class ActionDefinition:
    """An action, the parameters to solve for, and the constraints every legal action satisfies."""

    name: str
    variables: tuple[Variable, ...]
    constraints: tuple[Expression, ...]
