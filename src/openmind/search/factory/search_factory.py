from openmind.policy.factory.policy_factory import create_rated_policy_picker, create_solver_optimizer
from openmind.search.service.improvised import Improvised
from openmind.search.service.minimax import Minimax
from openmind.utility.factory.utility_factory import create_utility


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
