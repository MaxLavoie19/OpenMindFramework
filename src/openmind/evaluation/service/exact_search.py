import logging
import math

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.predictor.service.predictor import Predictor
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

logger = logging.getLogger(__name__)


class ExactSearch:
    """Perfect play by searching every reachable state; each state's value is computed once per domain."""

    def __init__(self, solver: Solver, predictor: Predictor, state_reader: StateReader) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._values: dict[tuple[str, State], tuple[float, ...]] = {}

    def optimal_actions(self, domain: Domain, state: State) -> tuple[Action, ...]:
        """The legal actions with the highest expected payoff for the player to act, in the solver's order."""
        actions = self._solver.solve(domain.problem, state)
        if not actions:
            raise ValueError("No legal action in this state")
        player = self._state_reader.player_to_act(state, domain.players)
        values = [self._action_value(domain, state, action)[player] for action in actions]
        best = max(values)
        return tuple(action for action, value in zip(actions, values) if math.isclose(value, best))

    def positions(self, domain: Domain) -> tuple[State, ...]:
        """Every state reachable from the initial state that has a legal action."""
        seen: set[State] = set()
        positions: list[State] = []
        pending = [domain.initial_state]
        while pending:
            state = pending.pop()
            if state in seen:
                continue
            seen.add(state)
            actions = self._solver.solve(domain.problem, state)
            if actions:
                positions.append(state)
            for action in actions:
                outcomes = self._predictor.predict(domain.transitions, state, action).outcomes
                pending.extend(outcome for outcome, _ in outcomes)
        logger.info("%s has %d positions with a legal action", domain.name, len(positions))
        return tuple(positions)

    def _value(self, domain: Domain, state: State) -> tuple[float, ...]:
        key = (domain.name, state)
        if key not in self._values:
            actions = self._solver.solve(domain.problem, state)
            if actions:
                player = self._state_reader.player_to_act(state, domain.players)
                self._values[key] = max(
                    (self._action_value(domain, state, action) for action in actions),
                    key=lambda value: value[player],
                )
            else:
                self._values[key] = self._state_reader.payoffs(state, domain.players)
        return self._values[key]

    def _action_value(self, domain: Domain, state: State, action: Action) -> tuple[float, ...]:
        outcomes = [
            (self._value(domain, outcome), probability)
            for outcome, probability in self._predictor.predict(domain.transitions, state, action).outcomes
        ]
        return tuple(
            math.fsum(probability * value[index] for value, probability in outcomes)
            for index in range(len(domain.players.names))
        )
