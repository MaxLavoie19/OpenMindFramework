from collections.abc import Mapping

from openmind.structure.model.grid import Grid
from openmind.structure.model.list import List
from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar
from openmind.world.model.state import State


class StateNamespaceMapper:
    """Maps a state to the names a rule reads, and a script's names back to a state.

    Every model is a name: a scalar reads as its value, so `turn == "X"`; a grid, a list or a map reads as itself, with
    its methods, so `cell[2, 3]` and `cell.lines(3)`. A script writes a model by assigning its name: a value to a
    scalar, a new grid, list or map to the others, such as `cell = cell.placed((row, col), turn)`."""

    def to_namespace(self, state: State) -> dict[str, object]:
        return {name: model.value if isinstance(model, Scalar) else model for name, model in state.models}

    def to_namespace_after(self, before: State, namespace: Mapping[str, object], state: State) -> dict[str, object]:
        """The namespace for a state reached from `before`: models are immutable, so it is built afresh."""
        return self.to_namespace(state)

    def to_state(self, state: State, namespace: Mapping[str, object]) -> State:
        """The state's models as the script left them. A scalar takes the value its name holds; any other model must be
        left as a model of its kind, or ValueError is raised."""
        models = []
        for name, model in state.models:
            written = namespace[name]
            if isinstance(model, Scalar):
                if isinstance(written, Scalar | Grid | List | Map):
                    raise ValueError(f"{name} is a Scalar; a script writes it a value, not a {type(written).__name__}")
                models.append((name, model if written == model.value else Scalar(written)))  # type: ignore[arg-type]
            elif isinstance(written, type(model)):
                models.append((name, written))
            else:
                raise ValueError(f"{name} is a {type(model).__name__}; a script left it {written!r}")
        return State(tuple(models))
