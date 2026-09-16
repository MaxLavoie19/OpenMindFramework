from typing import Protocol

from openmind.agent.model.domain import Domain
from openmind.timing.model.clock import Clock
from openmind.world.model.action import Action
from openmind.world.model.state import State


class Policy(Protocol):
    """Anything that chooses an action in a domain's state; where players act at once, for the player it's given. On a
    clock, it's given the player's clock and the steps that player has played so far in the game."""

    def choose(
        self, domain: Domain, state: State, player: str | None = None, clock: Clock | None = None, steps_played: int = 0
    ) -> Action: ...
