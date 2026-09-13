from dataclasses import dataclass

from openmind.expression.model.expression import Expression
from openmind.expression.model.state_variable import StateVariable


@dataclass(frozen=True, slots=True)
class Assign:
    """Sets an existing state variable to the value of an expression."""

    target: StateVariable
    value: Expression
