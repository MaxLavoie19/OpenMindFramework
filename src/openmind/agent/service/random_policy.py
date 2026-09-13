import random

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.world.model.action import Action
from openmind.world.model.state import State


class RandomPolicy:
    """Chooses uniformly among the legal actions."""

    def __init__(self, solver: Solver, rng: random.Random) -> None:
        self._solver = solver
        self._rng = rng

    def choose(self, domain: Domain, state: State) -> Action:
        actions = self._solver.solve(domain.problem, state)
        if not actions:
            raise ValueError("No legal action to choose from")
        return self._rng.choice(actions)
