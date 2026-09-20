from typing import Protocol

from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.value import Value
from openmind.world.model.state import State


class Reaching(Protocol):
    """What a player could reach in a position, one action ahead.

    A rule that forbids an action for what the other side could then do — leaving a king where it can be taken — is
    about the position the action leads to, not about the action. Reading it means asking the game what each player
    can do there, and where those actions land."""

    def players(self, state: State) -> tuple[Value, ...]:
        """Who could act in that position."""
        ...

    def cells(self, state: State, player: Value) -> tuple[Coordinates, ...]:
        """The cells that player's actions land on."""
        ...
