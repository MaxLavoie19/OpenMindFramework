from dataclasses import replace

from openmind.agent.model.domain import Domain
from openmind.agent.service.deduction_fallback import DeductionFallback
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.model.theory_of_mind import TheoryOfMind
from openmind.mcts.service.semi_determinized_search import SemiDeterminizedSearch
from openmind.mcts.service.tree_search import TreeSearch
from openmind.timing.model.clock import Clock
from openmind.timing.model.time_budget_estimator import TimeBudgetEstimator
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


class Agent:
    """Chooses actions in a domain by searching with MCTS, guided by a rater and valuing positions when it has models to.
    In a domain with an observation, it searches from what the player to act sees of the state it's given; with a theory
    of mind, it searches semi-determinized, once per hypothesis its theory gives. With a deduction fallback, a position
    its rules have no clue about is deduced first, and a proven choice is made without searching. Where players act at
    once, it searches for the player it's given, the other players playing the strategies its theory of mind predicts,
    if any, and samples its action from its average strategy. Given a clock, it asks its time budget estimator how long
    the step may take and searches for that long, any iterations it was built with kept as a cap; without one, it
    searches its iterations."""

    def __init__(
        self,
        tree_search: TreeSearch,
        settings: SearchSettings,
        guidance: Guidance | None = None,
        valuation: LeafValuation | None = None,
        fallback: DeductionFallback | None = None,
        semi_determinized_search: SemiDeterminizedSearch | None = None,
        theory: TheoryOfMind | None = None,
        state_reader: StateReader | None = None,
        estimator: TimeBudgetEstimator | None = None,
    ) -> None:
        self._tree_search = tree_search
        self._settings = settings
        self._guidance = guidance
        self._valuation = valuation
        self._fallback = fallback
        self._semi_determinized_search = semi_determinized_search
        self._theory = theory
        self._state_reader = StateReader() if state_reader is None else state_reader
        self._estimator = estimator

    def search(
        self, domain: Domain, state: State, player: str | None = None, clock: Clock | None = None, steps_played: int = 0
    ) -> SearchResult:
        """`player` names the searching player where players act at once; elsewhere it's the player to act. A clock
        without an estimator, or no clock for an agent built without iterations, raise ValueError."""
        settings = self._settings_for(clock, steps_played)
        result = self._search(domain, state, player, settings)
        return result if clock is None else replace(result, budget=settings.seconds)

    def _search(self, domain: Domain, state: State, player: str | None, settings: SearchSettings) -> SearchResult:
        if self._state_reader.acts_at_once(state, domain.players):
            return self._tree_search.search(
                domain.problem,
                domain.transitions,
                domain.players,
                state,
                settings,
                self._guidance,
                self._valuation,
                domain.observation,
                None,
                player,
                self._predicted(domain, state, player),
            )
        if self._fallback is not None:
            deduced = self._fallback.result(domain, state, self._valuation)
            if deduced is not None:
                return deduced
        if domain.observation is not None and self._semi_determinized_search is not None and self._theory is not None:
            return self._semi_determinized_search.search(
                domain, state, settings, self._theory, self._guidance, self._valuation
            )
        return self._tree_search.search(
            domain.problem,
            domain.transitions,
            domain.players,
            state,
            settings,
            self._guidance,
            self._valuation,
            domain.observation,
        )

    def choose(
        self, domain: Domain, state: State, player: str | None = None, clock: Clock | None = None, steps_played: int = 0
    ) -> Action:
        return self.search(domain, state, player, clock, steps_played).chosen

    def _settings_for(self, clock: Clock | None, steps_played: int) -> SearchSettings:
        """The settings a step searches with: on a clock, the estimator's budget as seconds."""
        if clock is None:
            if self._settings.iterations is None:
                raise ValueError("An agent built without iterations needs a clock to know how long to search")
            return self._settings
        if self._estimator is None:
            raise ValueError("An agent given a clock needs a time budget estimator")
        return replace(self._settings, seconds=self._estimator.budget(clock, steps_played))

    def _predicted(
        self, domain: Domain, state: State, player: str | None
    ) -> dict[str, tuple[tuple[Action, float], ...]] | None:
        """The strategies the theory of mind predicts for the other players to act; None without a theory or any
        prediction."""
        if self._theory is None or player is None:
            return None
        predicted: dict[str, tuple[tuple[Action, float], ...]] = {}
        for index in self._state_reader.players_to_act(state, domain.players):
            other = domain.players.names[index]
            if other != player and (strategy := self._theory.strategy(domain, state, player, other)) is not None:
                predicted[other] = strategy
        return predicted or None
