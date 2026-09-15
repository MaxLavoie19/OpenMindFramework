from dataclasses import dataclass

from openmind.world.model.action import Action


@dataclass(frozen=True, slots=True)
class JointAction:
    """The actions players take at once: each acting player's name with its action, in the order of the players' names."""

    actions: tuple[tuple[str, Action], ...]
