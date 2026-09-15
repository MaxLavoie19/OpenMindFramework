from typing import Protocol

from openmind.agent.model.domain import Domain
from openmind.world.model.action import Action
from openmind.world.model.state import State


class Policy(Protocol):
    """Anything that chooses an action in a domain's state; where players act at once, for the player it's given."""

    def choose(self, domain: Domain, state: State, player: str | None = None) -> Action: ...
