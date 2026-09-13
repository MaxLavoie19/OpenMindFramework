import pytest

from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain


@pytest.mark.log_level("INFO")
def test_create_agent_chooses_a_tictactoe_action() -> None:
    domain = create_tictactoe_domain()

    action = create_agent(iterations=20, seed=1).choose(domain, domain.initial_state)

    assert action.name == "place"
