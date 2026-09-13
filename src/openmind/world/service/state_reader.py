from openmind.world.model.state import State
from openmind.world.model.value import Value


class StateReader:
    """Reads a state variable's value by name."""

    def value(self, state: State, name: str) -> Value:
        for key, value in state.variables:
            if key == name:
                return value
        raise KeyError(f"Unknown state variable: {name!r}")
