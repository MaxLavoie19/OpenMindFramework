import random

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.world.model.action import Action
from openmind.world.model.state import State


class RandomPolicy:
    """Chooses uniformly among the legal actions; where players act at once, among the given player's."""

    def __init__(self, solver: Solver, rng: random.Random) -> None:
        self._solver = solver
        self._rng = rng

    def choose(self, domain: Domain, state: State, player: str | None = None) -> Action:
        actions = self._solver.solve(domain.problem, state, player=player)
        if not actions:
            raise ValueError("No legal action to choose from")
        return self._rng.choice(actions)
