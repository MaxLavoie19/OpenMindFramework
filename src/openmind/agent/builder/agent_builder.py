from typing import Self

from openmind.agent.constant.agent_constant import PRIOR_WEIGHT, ROLLOUT_TEMPERATURE
from openmind.agent.service.agent import Agent
from openmind.csp.service.solver import Solver
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.mcts.model.action_rater import ActionRater
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class AgentBuilder:
    """Sets how an agent searches, and what guides it, and wires the services it searches with."""

    def __init__(self) -> None:
        self._iterations: int | None = None
        self._exploration: float | None = None
        self._seed: int | None = None
        self._rater: ActionRater | None = None

    def with_iterations(self, iterations: int) -> Self:
        self._iterations = iterations
        return self

    def with_exploration(self, exploration: float) -> Self:
        self._exploration = exploration
        return self

    def with_seed(self, seed: int | None) -> Self:
        self._seed = seed
        return self

    def with_guidance(self, rater: ActionRater | None) -> Self:
        self._rater = rater
        return self

    def build(self) -> Agent:
        iterations, exploration = self._iterations, self._exploration
        if iterations is None or exploration is None:
            raise ValueError("Agent needs iterations and exploration")
        if iterations < 1:
            raise ValueError(f"Agent needs at least 1 iteration, not {iterations}")
        names = VariableNameMapper()
        interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
        tree_search = TreeSearch(
            Solver(interpreter, expression_text, action_text),
            Predictor(interpreter, names, expression_text, action_text),
            StateReader(),
            action_text,
        )
        guidance = Guidance(self._rater, PRIOR_WEIGHT, ROLLOUT_TEMPERATURE) if self._rater is not None else None
        return Agent(tree_search, SearchSettings(iterations, exploration, self._seed), guidance)
