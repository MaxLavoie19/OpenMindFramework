import json
from typing import Self

from openmind.agent.constant.agent_constant import GUIDED_ROLLOUTS, NOT_REBUILDABLE, PRIOR_WEIGHT, ROLLOUT_TEMPERATURE
from openmind.agent.model.model_description import ModelDescription
from openmind.agent.service.agent import Agent
from openmind.agent.service.deduction_fallback import DeductionFallback
from openmind.agent.service.one_ply_chooser import OnePlyChooser
from openmind.inference.model.deduction_budget import DeductionBudget
from openmind.inference.service.position_deducer import PositionDeducer
from openmind.mcts.constant.mcts_constant import DEFAULT_PUCT_EXPLORATION, UCB1
from openmind.mcts.model.action_rater import ActionRater
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.move_prior import MovePrior
from openmind.mcts.model.position_valuer import PositionValuer
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.model.theory_of_mind import TheoryOfMind
from openmind.mcts.service.semi_determinized_search import SemiDeterminizedSearch
from openmind.mcts.service.tree_search import TreeSearch
from openmind.rbs.factory.rule_factory import create_rule_caller
from openmind.timing.model.time_budget_estimator import TimeBudgetEstimator
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
        self._semi_determinized = False
        self._theory: TheoryOfMind | None = None
        self._estimator: TimeBudgetEstimator | None = None
        self._selection = UCB1
        self._puct_exploration = DEFAULT_PUCT_EXPLORATION
        self._prior: MovePrior | None = None

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

    def with_theory_of_mind(self, theory: TheoryOfMind) -> Self:
        """Searches semi-determinized, over the hypotheses of the given theory of mind: what the agent itself believes
        the position may be. Without one, the agent searches the position it is given."""
        self._semi_determinized = True
        self._theory = theory
        return self

    def with_time_budget_estimator(self, estimator: TimeBudgetEstimator | None) -> Self:
        """How the agent budgets a step's time when it's given a clock; with one, iterations become a cap and may be left
        out, the agent then searching on time alone. None: the agent can't play on a clock."""
        self._estimator = estimator
        return self

    def with_selection(self, selection: str, puct_exploration: float = DEFAULT_PUCT_EXPLORATION) -> Self:
        """How a tried node picks the action to follow: `ucb1`, the default, or `puct` with its exploration weight."""
        self._selection = selection
        self._puct_exploration = puct_exploration
        return self

    def with_prior(self, prior: MovePrior | None) -> Self:
        """The prior PUCT follows; None, the default, has every action alike."""
        self._prior = prior
        return self

    def describe(self, name: str) -> ModelDescription:
        """The agent this builder builds, under that name, as JSON text: every setting but the seed, which changes from
        game to game, and every model it plays with as the model describes itself; a model that can't is written as its
        class, marked as not rebuildable."""
        deduction = self._deduction
        text = json.dumps(
            {
                "iterations": self._iterations,
                "exploration": self._exploration,
                "guidance": _described(self._rater),
                "guided_rollouts": self._guided_rollouts,
                "valuation": _described(self._valuer),
                "rollout_actions": self._rollout_actions,
                "rollout_limit": self._rollout_limit,
                "unfinished_payoff": self._unfinished_payoff,
                "deduction": None
                if deduction is None
                else {"plies": deduction.plies, "seconds": deduction.seconds, "highest": deduction.highest},
                "theory_of_mind": _described(self._theory) if self._semi_determinized else None,
                "semi_determinized": self._semi_determinized,
                "time_budget_estimator": _described(self._estimator),
                "selection": self._selection,
                "puct_exploration": self._puct_exploration,
                "prior": _described(self._prior),
            },
            sort_keys=True,
        )
        return ModelDescription(name, text)

    def build(self) -> Agent:
        iterations, exploration = self._iterations, self._exploration
        if exploration is None:
            raise ValueError("Agent needs exploration")
        if iterations is None and self._estimator is None:
            raise ValueError("Agent needs iterations, a time budget estimator or both")
        if iterations is not None and iterations < 1:
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
        state_reader, action_text_mapper = StateReader(), ActionTextMapper()
        tree_search = TreeSearch(state_reader, action_text_mapper)
        fallback = (
            None
            if deduction is None
            else DeductionFallback(PositionDeducer(state_reader, action_text_mapper), state_reader, deduction)
        )
        guidance = (
            Guidance(self._rater, PRIOR_WEIGHT, ROLLOUT_TEMPERATURE, self._guided_rollouts)
            if self._rater is not None
            else None
        )
        valuation = LeafValuation(self._valuer, self._rollout_actions) if self._valuer is not None else None
        settings = SearchSettings(
            iterations,
            exploration,
            self._seed,
            self._rollout_limit,
            self._unfinished_payoff,
            selection=self._selection,
            puct_exploration=self._puct_exploration,
            prior=self._prior,
        )
        if not self._semi_determinized:
            return Agent(
                tree_search,
                settings,
                guidance,
                valuation,
                fallback,
                estimator=self._estimator,
                one_ply=OnePlyChooser(state_reader),
            )
        theory = self._theory
        if theory is None:
            raise ValueError("A semi-determinized agent needs a theory of mind: nothing else says what it can't see")
        semi_determinized = SemiDeterminizedSearch(tree_search, state_reader, action_text_mapper)
        return Agent(
            tree_search,
            settings,
            guidance,
            valuation,
            fallback,
            semi_determinized,
            theory,
            estimator=self._estimator,
            one_ply=OnePlyChooser(state_reader),
        )


def _described(part: object | None) -> object:
    """A part as it describes itself, read back as JSON; its class, marked as not rebuildable, when it can't; None
    without one."""
    if part is None:
        return None
    describe = getattr(part, "describe", None)
    if callable(describe):
        return json.loads(describe())
    kind = type(part)
    return {"class": f"{kind.__module__}.{kind.__qualname__}", NOT_REBUILDABLE: True}
