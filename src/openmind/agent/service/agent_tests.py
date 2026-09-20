from collections.abc import Callable

from openmind.agent.factory.agent_factory import create_actor, create_agent
from openmind.agent.service.actor_tests import Dispatched
from openmind.budget.model.budget import Budget
from openmind.knowledge.constant.task_constant import PLANNING, POSITION_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.factory.search_factory import create_minimax
from openmind.search.model.guidance import Guidance
from openmind.world.service.world import World

type Game = Callable[[str], RuleBasedGame]


def test_the_agent_strategizes_and_its_actor_dispatches_what_it_worked_out(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    world = World(played.start())
    dispatched = Dispatched()
    agent = create_agent(create_minimax(), create_actor(dispatched))

    strategy = agent.play(knowledge, played, world, Guidance("X"), Budget(5.0))

    assert strategy is not None and strategy.chosen(world.current()) is not None
    assert dispatched.actions == [strategy.chosen(world.current())]


def test_an_agent_with_nothing_to_play_here_dispatches_nothing_and_strategizes_for_its_turns(
    game: Game, knowledge: KnowledgeBase
) -> None:
    """It isn't O's turn, so nothing is dispatched; the strategy still covers the states where O will act."""
    played = game("tictactoe")
    world = World(played.start())
    dispatched = Dispatched()
    agent = create_agent(create_minimax(), create_actor(dispatched))

    strategy = agent.play(knowledge, played, world, Guidance("O"), Budget(5.0))

    assert dispatched.actions == []
    assert strategy is not None and strategy.chosen(world.current()) is None
    assert any(state.value("turn") == "O" for state in strategy.states())


def test_what_a_step_runs_with_is_the_time_management_policy_s(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    world = World(played.start())

    allocation = create_agent().allocate(knowledge, played, world, Budget(2.0), (PLANNING, POSITION_VALUE))

    assert allocation.settings.nodes > 0
    assert allocation.of(PLANNING) is None  # no planner is registered as a model yet
