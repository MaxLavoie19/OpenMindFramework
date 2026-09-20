from openmind.agent.model.dispatcher import Dispatcher
from openmind.agent.service.actor import Actor
from openmind.agent.service.agent import Agent
from openmind.budget.factory.budget_factory import create_plain_time_manager
from openmind.search.factory.search_factory import create_improvised
from openmind.search.model.planner import Planner


def create_actor(dispatcher: Dispatcher) -> Actor:
    """The actor performing what a strategy calls for, through that dispatcher."""
    return Actor(dispatcher)


def create_agent(planner: Planner[object] | None = None, actor: Actor | None = None) -> Agent:
    """An agent with the bootstrap time management model and the planner given, improvising where none is."""
    return Agent(create_plain_time_manager(), create_improvised() if planner is None else planner, actor)
