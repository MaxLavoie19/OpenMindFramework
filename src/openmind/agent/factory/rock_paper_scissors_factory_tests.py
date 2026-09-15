import pytest

from openmind.agent.factory.domain_factory import create_domain
from openmind.agent.factory.rock_paper_scissors_factory import create_rock_paper_scissors_domain
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.service.state_reader import StateReader


def throw(shape: str) -> Action:
    return Action("throw", (("shape", shape),))


def test_both_players_are_to_act_at_once_from_the_start() -> None:
    domain = create_rock_paper_scissors_domain()

    assert create_domain("rockpaperscissors") == domain
    assert StateReader().players_to_act(domain.initial_state, domain.players) == (0, 1)


@pytest.mark.parametrize(
    ("first", "second", "payoffs"),
    [("rock", "scissors", (1.0, 0.0)), ("rock", "rock", (0.5, 0.5)), ("rock", "paper", (0.0, 1.0)), ("scissors", "paper", (1.0, 0.0))],
)
def test_the_hands_thrown_at_once_decide_the_payoffs_and_end_the_game(
    first: str, second: str, payoffs: tuple[float, float]
) -> None:
    domain = create_rock_paper_scissors_domain()
    joint = JointAction((("A", throw(first)), ("B", throw(second))))

    ((state, probability),) = create_predictor().predict_joint(domain.transitions, domain.initial_state, joint).outcomes

    variables = dict(state.variables)
    assert probability == 1.0
    assert (variables["hand(A)"], variables["hand(B)"]) == (first, second)
    assert StateReader().payoffs(state, domain.players) == payoffs
    assert StateReader().players_to_act(state, domain.players) == ()
