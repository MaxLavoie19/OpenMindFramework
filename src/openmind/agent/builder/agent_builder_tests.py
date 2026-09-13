import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.factory.tictactoe_factory import create_tictactoe_domain
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.action import Action


@pytest.mark.log_level("INFO")
def test_build_gives_an_agent_that_searches() -> None:
    domain = create_tictactoe_domain()
    builder = StateBuilder()
    x_can_win = {"cell(1,1)": "X", "cell(1,2)": "X", "cell(2,1)": "O", "cell(2,2)": "O"}
    for name, value in (dict(domain.initial_state.variables) | x_can_win).items():
        builder.with_variable(name, value)

    agent = AgentBuilder().with_iterations(100).with_exploration(1.4).with_seed(1).build()

    assert agent.choose(domain, builder.build()) == Action("place", (("col", 3), ("row", 1)))


def test_build_rejects_missing_settings() -> None:
    with pytest.raises(ValueError, match="iterations"):
        AgentBuilder().with_exploration(1.4).build()


def test_build_rejects_fewer_than_one_iteration() -> None:
    with pytest.raises(ValueError, match="0"):
        AgentBuilder().with_iterations(0).with_exploration(1.4).build()
