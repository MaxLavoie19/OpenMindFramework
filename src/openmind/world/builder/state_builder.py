from operator import itemgetter
from typing import Self

from openmind.world.model.state import State
from openmind.world.model.value import Value


class StateBuilder:
    """Collects named variables into a state."""

    def __init__(self) -> None:
        self._variables: dict[str, Value] = {}

    def with_variable(self, name: str, value: Value) -> Self:
        if name in self._variables:
            raise ValueError(f"State variable {name!r} is already set")
        self._variables[name] = value
        return self

    def build(self) -> State:
        return State(tuple(sorted(self._variables.items(), key=itemgetter(0))))
