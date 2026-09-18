import math
from dataclasses import replace

from openmind.agent.service.one_ply_chooser import OnePlyChooser
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.mcts.model.action_statistics import ActionStatistics
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.search_result import SearchResult
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.timing.model.deadline import Deadline
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class DeductionFallback:
    """What an agent falls back on when its value rules have no clue: when the agent
    has no valuer, or its valuer can't value every legal action's outcomes or values them all the same for the player to
    act, it deduces the position within the budget; a proven best action is chosen without searching. The result holds
    that action alone, with one visit and its proven payoff for the player to act, and no samples."""

    def __init__(
        self,
        position_deducer: PositionDeducer,
        state_reader: StateReader,
        budget: DeductionBudget,
        one_ply: OnePlyChooser | None = None,
    ) -> None:
        self._one_ply = OnePlyChooser(state_reader) if one_ply is None else one_ply
        self._position_deducer = position_deducer
        self._state_reader = state_reader
        self._budget = budget

    def result(
        self, rbs: RuleBasedSystem, state: State, valuation: LeafValuation | None, deadline: Deadline | None = None
    ) -> SearchResult | None:
        """The proven choice, or None when the rules have a clue or nothing was proven. With a deadline, valuing the legal
        moves stops once it passes, the rules then taken to have a clue, and the deduction gets no more than the time
        left."""
        actions = rbs.actions(state)
        if not actions:
            return None
        player = self._state_reader.player_to_act(state, rbs.players())
        if valuation is not None:
            values = self._one_ply.values(rbs, state, actions, valuation.valuer, deadline)
            if values is None and deadline is not None and deadline.passed():
                return None
            if values is not None and not all(math.isclose(value, values[0]) for value in values):
                return None
        budget = self._budget
        if deadline is not None:
            left = deadline.remaining()
            if left <= 0.0:
                return None
            budget = replace(budget, seconds=min(budget.seconds, left))
        deduction = self._position_deducer.deduce(rbs, state, budget)
        if deduction.action is None or deduction.payoffs is None:
            return None
        statistics = (ActionStatistics(deduction.action, 1, deduction.payoffs[player]),)
        return SearchResult(rbs.players().names[player], statistics, deduction.action, ())
