import pytest

from openmind.agent.factory.rock_paper_scissors_factory import create_rock_paper_scissors_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.service.joint_solver import JointSolver
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

THROWS = tuple(Action("throw", (("shape", shape),)) for shape in ("rock", "paper", "scissors"))


def new_joint_solver() -> JointSolver:
    return JointSolver(create_solver(), StateReader())


def test_every_player_to_act_gets_its_legal_actions() -> None:
    domain = create_rock_paper_scissors_domain()

    assert new_joint_solver().legal(domain.problem, domain.initial_state, domain.players) == ((0, THROWS), (1, THROWS))


def test_a_state_where_no_player_is_to_act_is_over() -> None:
    domain = create_rock_paper_scissors_domain()
    joint = JointAction((("A", THROWS[0]), ("B", THROWS[1])))
    ((over, _),) = create_predictor().predict_joint(domain.transitions, domain.initial_state, joint).outcomes

    assert new_joint_solver().legal(domain.problem, over, domain.players) == ()


def test_a_player_to_act_without_an_action_while_another_has_one_raises() -> None:
    domain = create_rock_paper_scissors_domain()
    thrown = State(tuple((name, "rock" if name == "hand(A)" else value) for name, value in domain.initial_state.variables))

    with pytest.raises(ValueError, match="A can't act while other players to act can"):
        new_joint_solver().legal(domain.problem, thrown, domain.players)


def test_a_state_variable_named_player_raises_when_solving_for_a_player() -> None:
    domain = create_rock_paper_scissors_domain()
    clashing = State((*domain.initial_state.variables, ("player", "A")))

    with pytest.raises(ValueError, match="named 'player'"):
        new_joint_solver().legal(domain.problem, clashing, domain.players)
