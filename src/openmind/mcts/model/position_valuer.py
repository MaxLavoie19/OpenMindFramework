from typing import Protocol

from openmind.world.model.state import State


class PositionValuer(Protocol):
    """A model valuing positions: each player's expected payoff, in the order of the players' names, or None where it
    knows nothing."""

    def value(self, state: State) -> tuple[float, ...] | None: ...
