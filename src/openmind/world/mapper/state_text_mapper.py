from openmind.structure.model.scalar import Scalar
from openmind.world.model.state import State


class StateTextMapper:
    """Maps a state to readable text: one "name = model" line per model, a scalar written as its value."""

    def to_text(self, state: State) -> str:
        return "\n".join(
            f"{name} = {model.value!r}" if isinstance(model, Scalar) else f"{name} = {model!r}" for name, model in state.models
        )
