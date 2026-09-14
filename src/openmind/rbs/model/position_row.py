from dataclasses import dataclass

from openmind.world.model.state import State


@dataclass(frozen=True, slots=True)
class PositionRow:
    """A position valued for a player, by name, and the payoff its value is fitted to."""

    state: State
    player: str
    target: float
