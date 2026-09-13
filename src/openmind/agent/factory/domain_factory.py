from openmind.agent.constant.tictactoe_constant import NAME as TICTACTOE
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.agent.model.domain import Domain


def create_domain(name: str) -> Domain:
    """Creates a domain within the agent from its name."""
    if name == TICTACTOE:
        return create_tictactoe_domain()
    raise ValueError(f"Unknown domain {name!r}; known domains: {TICTACTOE}")
