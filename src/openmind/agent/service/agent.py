from openmind.agent.model.domain import Domain
from openmind.agent.service.deduction_fallback import DeductionFallback
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.leaf_valuation import LeafValuation
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.world.model.action import Action
from openmind.world.model.state import State


class Agent:
    """Chooses actions in a domain by searching with MCTS, guided by a rater and valuing positions when it has models to.
    In a domain with an observation, it searches from what the player to act sees of the state it's given. With a
    deduction fallback, a position its rules have no clue about is deduced first, and a proven choice is made without
    searching."""

    def __init__(
        self,
        tree_search: TreeSearch,
        settings: SearchSettings,
        guidance: Guidance | None = None,
        valuation: LeafValuation | None = None,
        fallback: DeductionFallback | None = None,
    ) -> None:
        self._tree_search = tree_search
        self._settings = settings
        self._guidance = guidance
        self._valuation = valuation
        self._fallback = fallback

    def search(self, domain: Domain, state: State) -> SearchResult:
        if self._fallback is not None:
            deduced = self._fallback.result(domain, state, self._valuation)
            if deduced is not None:
                return deduced
        return self._tree_search.search(
            domain.problem,
            domain.transitions,
            domain.players,
            state,
            self._settings,
            self._guidance,
            self._valuation,
            domain.observation,
        )

    def choose(self, domain: Domain, state: State) -> Action:
        return self.search(domain, state).chosen
