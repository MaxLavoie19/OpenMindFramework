import logging
import math
import time
from collections.abc import Sequence

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.evaluation.model.action_values import ActionValues
from openmind.evaluation.model.choice_measure import ChoiceMeasure
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.model.action import Action
from openmind.world.model.state import State

logger = logging.getLogger(__name__)


class ChoiceMeasurer:
    """Measures an agent's choices against the values of every legal action."""

    def __init__(self, state_text_mapper: StateTextMapper, action_text_mapper: ActionTextMapper) -> None:
        self._state_text_mapper = state_text_mapper
        self._action_text_mapper = action_text_mapper

    def measure(
        self,
        rbs: RuleBasedSystem,
        agent_builder: AgentBuilder,
        positions: Sequence[tuple[State, ActionValues]],
        tolerance: float,
        kind: str,
        iterations: int,
    ) -> tuple[ChoiceMeasure, ...]:
        """Builds the agent once and searches every position with it, in order; kind and iterations name the agent in
        the logs."""
        agent = agent_builder.build()
        measures: list[ChoiceMeasure] = []
        for state, action_values in positions:
            started = time.perf_counter()
            result = agent.search(rbs, state)
            seconds = time.perf_counter() - started
            value_of = dict(action_values)
            optimal_actions = self.optimal(action_values, tolerance)
            visits = sum(item.visits for item in result.statistics)
            optimal_visits = sum(item.visits for item in result.statistics if item.action in optimal_actions)
            share = optimal_visits / visits if visits else 0.0
            regret = max(value_of.values()) - value_of[result.chosen]
            measures.append(ChoiceMeasure(result.chosen in optimal_actions, share, regret, seconds))
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "%s at %d iterations: chose %s, regret %s; optimal: %s; %s of visits on optimal actions; state: %s",
                    kind,
                    iterations,
                    self._action_text_mapper.to_text(result.chosen),
                    regret,
                    ", ".join(self._action_text_mapper.to_text(action) for action, _ in action_values if action in optimal_actions),
                    share,
                    self._state_text_mapper.to_text(state).replace("\n", ", "),
                )
        return tuple(measures)

    def optimal(self, action_values: ActionValues, tolerance: float) -> frozenset[Action]:
        """The actions within tolerance of the best value; with a tolerance of 0, those equal to it."""
        best = max(value for _, value in action_values)
        if tolerance == 0.0:
            return frozenset(action for action, value in action_values if math.isclose(value, best))
        return frozenset(action for action, value in action_values if value >= best - tolerance)
