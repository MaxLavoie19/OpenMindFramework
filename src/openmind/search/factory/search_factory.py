from openmind.heuristic.service.successor_move_rater import SuccessorMoveRater
from openmind.policy.factory.policy_factory import create_rated_policy_picker, create_solver_optimizer
from openmind.search.repository.search_tree_cache import SearchTreeCache
from openmind.search.service.greedy import Greedy
from openmind.search.service.improvised import Improvised
from openmind.search.service.minimax import Minimax
from openmind.search.service.monte_carlo_tree_search import MonteCarloTreeSearch
from openmind.utility.factory.utility_factory import create_utility


def create_greedy(successor_move_rater: SuccessorMoveRater | None = None) -> Greedy:
    """The planner that looks one step: the best-rated action here, played. Built with what rates an action by
    what it leads to, which it asks through the position value the guidance carried rather than one of its
    own — so the rater it is given needs no valuer."""
    return Greedy(SuccessorMoveRater() if successor_move_rater is None else successor_move_rater)


def create_improvised(policy_picker: object | None = None, optimizer: object | None = None) -> Improvised:
    """The planner that doesn't plan: the picked policy's optimizer gives the move. Built once with the picker and the
    optimizer it uses, the rated picker and the solver optimizer unless others are given."""
    return Improvised(
        create_rated_policy_picker() if policy_picker is None else policy_picker,  # type: ignore[arg-type]
        create_solver_optimizer() if optimizer is None else optimizer,  # type: ignore[arg-type]
    )


def create_minimax() -> Minimax:
    """The planner reading a small game out to its end."""
    return Minimax(create_utility())


def create_monte_carlo_tree_search(trees: SearchTreeCache | None = None) -> MonteCarloTreeSearch:
    """The planner exploring the lines a position leads to, with the tree cache it keeps them in: one of its own
    unless another is given, so a search built once carries its trees from move to move."""
    return MonteCarloTreeSearch(create_utility(), SearchTreeCache() if trees is None else trees)
