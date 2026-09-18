from typing import Protocol

from openmind.world.model.state import State


class PositionValuer(Protocol):
    """A model valuing positions: each player's expected payoff, in the order of the players' names, or None where it
    knows nothing. An RBS fills this role with its position rules; a learned model can fill it instead."""

    def values(self, state: State) -> tuple[float, ...] | None: ...
