from typing import Protocol

from openmind.world.model.state import State


class PositionValuer[Model](Protocol):
    """The position value task: what a state is worth to each player, in the players' order, or None where the model
    knows nothing of it. A stateless service fills it, given the model it runs: a position value ruleset through the
    RBS, a lookup table, a network, …"""

    def values(self, model: Model, state: State) -> tuple[float, ...] | None: ...
