import math
import random
from collections.abc import Sequence

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.mcts.constant.mcts_constant import RANDOM_OPTION
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.position_valuer import PositionValuer
from openmind.mcts.model.search_result import SearchResult
from openmind.predictor.service.predictor import Predictor
from openmind.timing.model.deadline import Deadline
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class OnePlyChooser:
    """What a player choosing without searching needs: its legal moves, each legal move valued once (its outcomes valued
    for the player to act and weighed by their probabilities, a finished outcome at its payoffs), which the deduction
    fallback reads, and a random legal move, for a player out of time."""

    def __init__(self, solver: Solver, predictor: Predictor, state_reader: StateReader) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader

    def legal(self, domain: Domain, state: State) -> tuple[Action, ...]:
        return self._solver.solve(domain.problem, state)

    def values(
        self,
        domain: Domain,
        state: State,
        actions: Sequence[Action],
        valuer: PositionValuer,
        deadline: Deadline | None = None,
    ) -> list[float] | None:
        """Each action's value for the player to act, in the actions' order; None when the valuer can't value an outcome,
        or the deadline passed before every action was valued."""
        player = self._state_reader.player_to_act(state, domain.players)
        values: list[float] = []
        for action in actions:
            if deadline is not None and deadline.passed():
                return None
            worth: list[float] = []
            for outcome, probability in self._predictor.predict(domain.transitions, state, action).outcomes:
                if self._solver.solve(domain.problem, outcome):
                    valued = valuer.value(outcome)
                    if valued is None:
                        return None
                    worth.append(probability * valued[player])
                else:
                    worth.append(probability * self._state_reader.payoffs(outcome, domain.players)[player])
            values.append(math.fsum(worth))
        return values

    def random(self, domain: Domain, state: State, rng: random.Random, unknown: float) -> SearchResult:
        """A random legal move, with one visit at the value given for what isn't known. No legal move raises ValueError."""
        actions = self.legal(domain, state)
        if not actions:
            raise ValueError("No legal action to choose from")
        chosen = rng.choice(actions)
        player = domain.players.names[self._state_reader.player_to_act(state, domain.players)]
        return SearchResult(player, (ActionStatistics(chosen, 1, unknown),), chosen, (), option=RANDOM_OPTION)
