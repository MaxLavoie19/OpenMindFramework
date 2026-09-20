from collections.abc import Sequence

from openmind.abstraction.factory.abstraction_factory import create_rule_abstractor
from openmind.agent.model.dispatcher import Dispatcher
from openmind.agent.service.actor import Actor
from openmind.agent.service.agent import Agent
from openmind.agent.service.hierarchy import Hierarchy
from openmind.agent.service.level import Level
from openmind.budget.factory.budget_factory import create_plain_time_manager
from openmind.knowledge.constant.task_constant import ABSTRACTION
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.rbs.factory.rbs_factory import find_rule_based_system
from openmind.rbs.service.rule_based_game import RuleBasedGame
from openmind.search.factory.search_factory import create_improvised
from openmind.search.model.planner import Planner
from openmind.search.model.guidance import Guidance
from openmind.utility.factory.utility_factory import create_utility
from openmind.world.service.world import World


def create_actor(dispatcher: Dispatcher) -> Actor:
    """The actor performing what a strategy calls for, through that dispatcher."""
    return Actor(dispatcher)


def create_agent(planner: Planner[object] | None = None, actor: Actor | None = None) -> Agent:
    """An agent with the bootstrap time management model and the planner given, improvising where none is."""
    return Agent(create_plain_time_manager(), create_improvised() if planner is None else planner, actor)


def create_level(
    knowledge_base: KnowledgeBase,
    game: RuleBasedGame,
    guidance: Guidance,
    agent: Agent | None = None,
    world: World | None = None,
) -> Level:
    """A level of the hierarchy playing that game: its own world, starting where the game starts, and the abstraction
    ruleset of its context where it has one, so it sees what that ruleset says and the state as it is otherwise."""
    abstraction = find_rule_based_system(knowledge_base, game.context, ABSTRACTION)
    return Level(
        game.context_id,
        game,
        create_agent() if agent is None else agent,
        guidance,
        World(game.start()) if world is None else world,
        None if abstraction is None else (abstraction, create_rule_abstractor()),
        create_utility(),
    )


def create_hierarchy(levels: Sequence[Level] = ()) -> Hierarchy:
    """The levels OMF runs, each found by its context id."""
    return Hierarchy(levels)
