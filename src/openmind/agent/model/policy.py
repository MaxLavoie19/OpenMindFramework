from typing import Protocol

from openmind.agent.model.domain import Domain
from openmind.world.model.action import Action
from openmind.world.model.state import State


class Policy(Protocol):
    """Anything that chooses an action in a domain's state."""

    def choose(self, domain: Domain, state: State) -> Action: ...
