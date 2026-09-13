from openmind.world.model.action import Action


class ActionTextMapper:
    """Maps an action to readable text: place(col=3, row=2)."""

    def to_text(self, action: Action) -> str:
        parameters = ", ".join(f"{name}={value!r}" for name, value in action.parameters)
        return f"{action.name}({parameters})"
