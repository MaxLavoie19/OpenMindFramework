from collections.abc import Callable
import pytest

from openmind.rbs.service.rule_based_system import RuleBasedSystem

from openmind.agent.factory.agent_factory import create_agent

type Game = Callable[[str], RuleBasedSystem]


@pytest.mark.log_level("INFO")
def test_create_agent_chooses_a_tictactoe_action(game: Game) -> None:
    rbs = game("tictactoe")

    action = create_agent(iterations=20, seed=1).choose(rbs, rbs.start())

    assert action.name == "place"
