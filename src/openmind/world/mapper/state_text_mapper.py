from openmind.world.model.state import State


class StateTextMapper:
    """Maps a state to readable text: one "name = value" line per variable."""

    def to_text(self, state: State) -> str:
        return "\n".join(f"{name} = {value!r}" for name, value in state.variables)
