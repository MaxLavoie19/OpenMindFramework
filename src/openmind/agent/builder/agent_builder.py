from typing import Self

from openmind.agent.constant.agent_constant import GUIDED_ROLLOUTS, PRIOR_WEIGHT, ROLLOUT_TEMPERATURE
from openmind.agent.service.agent import Agent
from openmind.agent.service.deduction_fallback import DeductionFallback
from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.mcts.model.action_rater import ActionRater
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.position_valuer import PositionValuer
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.service.state_reader import StateReader


class AgentBuilder:
    """Sets how an agent searches, what guides it and what values its positions, and wires the services it searches
    with."""

    def __init__(self) -> None:
        self._iterations: int | None = None
        self._exploration: float | None = None
        self._seed: int | None = None
        self._rater: ActionRater | None = None
        self._guided_rollouts = GUIDED_ROLLOUTS
        self._valuer: PositionValuer | None = None
        self._rollout_actions = 0
        self._rollout_limit: int | None = None
        self._unfinished_payoff: float | None = None
        self._deduction: DeductionBudget | None = None

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

    def with_guided_rollouts(self, guided: bool) -> Self:
        """Whether a guided agent's rollouts follow the ratings; without, only its tree's nodes are rated."""
        self._guided_rollouts = guided
        return self

    def with_valuation(self, valuer: PositionValuer | None) -> Self:
        """The model valuing the positions the agent's rollouts reach, instead of playing them out; None plays them out."""
        self._valuer = valuer
        return self

    def with_rollout_actions(self, actions: int) -> Self:
        """How many rollout actions a valuing agent plays before valuing the position, 0 by default."""
        self._rollout_actions = actions
        return self

    def with_rollout_limit(self, limit: int | None, unfinished_payoff: float | None = None) -> Self:
        """How many actions a rollout plays at most before every player gets the unfinished payoff; None plays rollouts to
        the end."""
        self._rollout_limit = limit
        self._unfinished_payoff = unfinished_payoff
        return self

    def with_deduction(self, budget: DeductionBudget | None) -> Self:
        """The budget of the deduction the agent falls back on when its value rules have no clue; None never deduces."""
        self._deduction = budget
        return self

    def build(self) -> Agent:
        iterations, exploration = self._iterations, self._exploration
        if iterations is None or exploration is None:
            raise ValueError("Agent needs iterations and exploration")
        if iterations < 1:
            raise ValueError(f"Agent needs at least 1 iteration, not {iterations}")
        if self._rollout_actions < 0:
            raise ValueError(f"Rollout actions can't be negative, not {self._rollout_actions}")
        if self._rollout_limit is not None and self._rollout_limit < 0:
            raise ValueError(f"The rollout limit can't be negative, not {self._rollout_limit}")
        if self._rollout_limit is not None and self._unfinished_payoff is None:
            raise ValueError("A rollout limit needs an unfinished payoff")
        deduction = self._deduction
        if deduction is not None and (deduction.plies < 1 or deduction.seconds <= 0.0):
            raise ValueError(f"A deduction needs at least 1 ply and more than 0 seconds, not {deduction}")
        solver, predictor, state_reader, action_text_mapper = (
            SolverBuilder().build(),
            PredictorBuilder().build(),
            StateReader(),
            ActionTextMapper(),
        )
        tree_search = TreeSearch(solver, predictor, state_reader, action_text_mapper)
        fallback = (
            None
            if deduction is None
            else DeductionFallback(
                PositionDeducer(solver, predictor, state_reader, action_text_mapper), solver, predictor, state_reader, deduction
            )
        )
        guidance = (
            Guidance(self._rater, PRIOR_WEIGHT, ROLLOUT_TEMPERATURE, self._guided_rollouts)
            if self._rater is not None
            else None
        )
        valuation = LeafValuation(self._valuer, self._rollout_actions) if self._valuer is not None else None
        settings = SearchSettings(iterations, exploration, self._seed, self._rollout_limit, self._unfinished_payoff)
        return Agent(tree_search, settings, guidance, valuation, fallback)
