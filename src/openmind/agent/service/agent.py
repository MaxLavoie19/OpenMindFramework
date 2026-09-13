from openmind.agent.model.domain import Domain
from openmind.mcts.model.guidance import Guidance
from openmind.mcts.model.search_result import SearchResult
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.world.model.action import Action
from openmind.world.model.state import State


class Agent:
    """Chooses actions in a domain by searching with MCTS, guided by a rater when it has one."""

    def __init__(self, tree_search: TreeSearch, settings: SearchSettings, guidance: Guidance | None = None) -> None:
        self._tree_search = tree_search
        self._settings = settings
        self._guidance = guidance

    def search(self, domain: Domain, state: State) -> SearchResult:
        return self._tree_search.search(
            domain.problem, domain.transitions, domain.players, state, self._settings, self._guidance
        )

    def choose(self, domain: Domain, state: State) -> Action:
        return self.search(domain, state).chosen
