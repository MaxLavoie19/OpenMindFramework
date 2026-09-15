from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction


class ActionTextMapper:
    """Maps an action to readable text: place(col=3, row=2); and actions taken at once: A: throw(shape='rock'), B:
    throw(shape='paper')."""

    def to_text(self, action: Action) -> str:
        parameters = ", ".join(f"{name}={value!r}" for name, value in action.parameters)
        return f"{action.name}({parameters})"

    def joint_text(self, joint: JointAction) -> str:
        return ", ".join(f"{player}: {self.to_text(action)}" for player, action in joint.actions)
