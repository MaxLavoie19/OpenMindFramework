import math
from collections.abc import Sequence

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.position_valuer import PositionValuer
from openmind.mcts.model.search_result import SearchResult
from openmind.predictor.service.predictor import Predictor
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class DeductionFallback:
    """What an agent falls back on when its value rules have no clue. In a domain every player sees whole, when the agent
    has no valuer, or its valuer can't value every legal action's outcomes or values them all the same for the player to
    act, it deduces the position within the budget; a proven best action is chosen without searching. The result holds
    that action alone, with one visit and its proven payoff for the player to act, and no samples."""

    def __init__(
        self,
        position_deducer: PositionDeducer,
        solver: Solver,
        predictor: Predictor,
        state_reader: StateReader,
        budget: DeductionBudget,
    ) -> None:
        self._position_deducer = position_deducer
        self._solver = solver
        self._predictor = predictor
        self._state_reader = state_reader
        self._budget = budget

    def result(self, domain: Domain, state: State, valuation: LeafValuation | None) -> SearchResult | None:
        """The proven choice, or None when the rules have a clue or nothing was proven."""
        if domain.observation is not None:
            return None
        actions = self._solver.solve(domain.problem, state)
        if not actions:
            return None
        player = self._state_reader.player_to_act(state, domain.players)
        if valuation is not None and self._distinguishes(domain, state, actions, player, valuation.valuer):
            return None
        deduction = self._position_deducer.deduce(domain, state, self._budget)
        if deduction.action is None or deduction.payoffs is None:
            return None
        statistics = (ActionStatistics(deduction.action, 1, deduction.payoffs[player]),)
        return SearchResult(domain.players.names[player], statistics, deduction.action, ())

    def _distinguishes(
        self, domain: Domain, state: State, actions: Sequence[Action], player: int, valuer: PositionValuer
    ) -> bool:
        values: list[float] = []
        for action in actions:
            worth: list[float] = []
            for outcome, probability in self._predictor.predict(domain.transitions, state, action).outcomes:
                if self._solver.solve(domain.problem, outcome):
                    valued = valuer.value(outcome)
                    if valued is None:
                        return False
                    worth.append(probability * valued[player])
                else:
                    worth.append(probability * self._state_reader.payoffs(outcome, domain.players)[player])
            values.append(math.fsum(worth))
        return not all(math.isclose(value, values[0]) for value in values)
