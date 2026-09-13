from typing import Self

from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.csp.model.variable import Variable
from openmind.expression.model.expression import Expression


class ProblemBuilder:
    """Collects action definitions into a constraint satisfaction problem."""

    def __init__(self) -> None:
        self._actions: list[ActionDefinition] = []

    def with_action(
        self, name: str, variables: tuple[Variable, ...], constraints: tuple[Expression, ...]
    ) -> Self:
        if any(action.name == name for action in self._actions):
            raise ValueError(f"Action {name!r} is already defined")
        variable_names = [variable.name for variable in variables]
        if len(set(variable_names)) != len(variable_names):
            raise ValueError(f"Action {name!r} repeats a variable name: {variable_names}")
        self._actions.append(ActionDefinition(name, variables, constraints))
        return self

    def build(self) -> Problem:
        return Problem(tuple(self._actions))
