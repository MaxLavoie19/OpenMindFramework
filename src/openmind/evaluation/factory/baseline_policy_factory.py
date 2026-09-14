import random

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.model.policy import Policy
from openmind.agent.service.random_policy import RandomPolicy
from openmind.csp.builder.solver_builder import SolverBuilder


def create_built_agent(agent_builder: AgentBuilder, seed: int) -> Policy:
    """The agent the builder builds. It keeps the builder's own seed, so it searches alike in every game."""
    return agent_builder.build()


def create_random_policy(seed: int) -> Policy:
    """A policy choosing uniformly among the legal actions, its choices drawn from the game's seed."""
    return RandomPolicy(SolverBuilder().build(), random.Random(seed))
