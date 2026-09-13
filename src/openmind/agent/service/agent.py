from openmind.agent.model.domain import Domain
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.tree_search import TreeSearch
from openmind.world.model.action import Action
from openmind.world.model.state import State


class Agent:
    """Chooses actions in a domain by searching with MCTS."""

    def __init__(self, tree_search: TreeSearch, settings: SearchSettings) -> None:
        self._tree_search = tree_search
        self._settings = settings

    def choose(self, domain: Domain, state: State) -> Action:
        result = self._tree_search.search(
            domain.problem, domain.transitions, domain.players, state, self._settings
        )
        return result.chosen
