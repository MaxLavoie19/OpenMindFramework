from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import DEFAULT_ITERATIONS, EXPLORATION
from openmind.agent.service.agent import Agent


def create_agent(iterations: int = DEFAULT_ITERATIONS, seed: int | None = None) -> Agent:
    """An MCTS agent searching with the UCT exploration weight √2."""
    return AgentBuilder().with_iterations(iterations).with_exploration(EXPLORATION).with_seed(seed).build()
