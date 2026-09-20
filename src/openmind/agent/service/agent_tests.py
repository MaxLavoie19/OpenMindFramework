from collections.abc import Callable

from openmind.agent.factory.agent_factory import create_actor, create_agent
from openmind.rule.model.python_rule import PythonRule
from openmind.agent.service.actor_tests import Dispatched
from openmind.budget.model.budget import Budget
from openmind.knowledge.constant.task_constant import PLANNING, POSITION_VALUE
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.factory.search_factory import create_minimax
from openmind.search.model.guidance import Guidance
from openmind.world.service.world import World

type Game = Callable[[str], RuleBasedGame]


class Remembers:
    """A planner of the test's own: it plans nothing and keeps what it was given to plan with."""

    def __init__(self) -> None:
        self.guidance: Guidance | None = None

    def plan(self, model, knowledge_base, node, guidance, settings):  # type: ignore[no-untyped-def]
        self.guidance = guidance
        return None


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


def test_the_agent_plays_with_the_models_the_policy_chose(
    game: Game, knowledge: KnowledgeBase, heuristic: Callable[..., object]
) -> None:
    """A heuristic deduced from the rules or learned from games is registered as a model of its task, and what reads
    it back is the agent: nothing has to hand it to the planner."""
    played = game("tictactoe")
    heuristic("tictactoe", "a constant", PythonRule("1.0"), 0.5)
    planner = Remembers()

    create_agent(planner).play(knowledge, played, World(played.start()), Guidance("X"), Budget(1.0))

    assert planner.guidance is not None and planner.guidance.position_value is not None
    model, valuer = planner.guidance.position_value
    assert valuer.values(model, played.node(played.start())) == (0.5, 0.5)  # the rule, at the weight it was linked at


def test_an_agent_with_no_heuristic_registered_plays_without_one(game: Game, knowledge: KnowledgeBase) -> None:
    played = game("tictactoe")
    planner = Remembers()

    create_agent(planner).play(knowledge, played, World(played.start()), Guidance("X"), Budget(1.0))

    assert planner.guidance is not None and planner.guidance.position_value is None
