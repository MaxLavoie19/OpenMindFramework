from collections.abc import Callable

from openmind.budget.factory.budget_factory import create_plain_time_manager
from openmind.budget.model.budget import Budget
from openmind.knowledge.constant.task_constant import MOVE_VALUE, POSITION_VALUE, SIMULATION
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.model.factory.model_factory import create_model_registry, create_model_timer
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.rule.model.python_rule import PythonRule

type Game = Callable[[str], RuleBasedGame]

TASKS = (SIMULATION, POSITION_VALUE, MOVE_VALUE)


def test_the_best_measured_model_of_each_task_is_what_a_step_runs_with(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    played = game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.5)
    node = played.node(played.start())

    allocation = create_plain_time_manager().manage(None, knowledge, node, Budget(1.0), TASKS)

    assert allocation.of(SIMULATION).name == "simulation"  # type: ignore[union-attr]
    assert allocation.of(POSITION_VALUE).name == POSITION_VALUE  # type: ignore[union-attr]
    assert allocation.of(MOVE_VALUE) is None


def test_more_seconds_buy_more_nodes_and_a_costlier_model_buys_fewer(
    game: Game, knowledge: KnowledgeBase
) -> None:
    played = game("tictactoe")
    node = played.node(played.start())
    manager, registry, timer = create_plain_time_manager(), create_model_registry(), create_model_timer()
    simulation = registry.of_task(knowledge, played.context_id, SIMULATION)[0]

    quick = manager.manage(None, knowledge, node, Budget(1.0), (SIMULATION,)).settings.nodes
    longer = manager.manage(None, knowledge, node, Budget(10.0), (SIMULATION,)).settings.nodes
    timer.spent(knowledge, simulation, 0.1)
    costly = manager.manage(None, knowledge, node, Budget(1.0), (SIMULATION,)).settings.nodes

    assert longer == quick * 10
    assert costly == 8  # a tenth of a second a node, of the 0.8 seconds the planner may spend


def test_a_step_with_no_seconds_still_gets_a_node(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")

    allocation = create_plain_time_manager().manage(None, knowledge, played.node(played.start()), Budget(0.0), TASKS)

    assert allocation.settings.nodes == 1
