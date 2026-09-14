from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import DEFAULT_ITERATIONS, EXPLORATION
from openmind.agent.service.agent import Agent


def create_agent(
    iterations: int = DEFAULT_ITERATIONS,
    seed: int | None = None,
    rollout_limit: int | None = None,
    unfinished_payoff: float | None = None,
) -> Agent:
    """An MCTS agent searching with the UCT exploration weight √2; with a rollout limit, a rollout still in play after
    that many actions gives every player the unfinished payoff."""
    return (
        AgentBuilder()
        .with_iterations(iterations)
        .with_exploration(EXPLORATION)
        .with_seed(seed)
        .with_rollout_limit(rollout_limit, unfinished_payoff)
        .build()
    )
