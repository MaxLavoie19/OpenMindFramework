from dataclasses import replace

import pytest

from openmind.agent.builder.agent_builder import AgentBuilder
from openmind.agent.constant.agent_constant import EXPLORATION
from openmind.agent.constant.prisoners_dilemma_constant import STANDARD
from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.factory.prisoners_dilemma_factory import create_prisoners_dilemma_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.model.action import Action
from openmind.world.model.joint_action import JointAction
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")

ONE_ROUND = replace(STANDARD, name="oneround", rounds=1)


def choose(choice: str) -> Action:
    return Action("choose", (("choice", choice),))


def after(*choices: str) -> State:
    domain, predictor = create_prisoners_dilemma_domain(ONE_ROUND), create_predictor()
    state = domain.initial_state
    for choice in choices:
        ((state, _),) = predictor.predict(domain.transitions, state, choose(choice)).outcomes
    return state


def test_in_one_round_the_agent_defects_as_either_player() -> None:
    domain = create_prisoners_dilemma_domain(ONE_ROUND)

    assert create_agent(300, seed=1).choose(domain, domain.initial_state) == choose("defect")
    assert create_agent(300, seed=1).choose(domain, after("cooperate")) == choose("defect")


def test_the_second_player_s_search_is_the_same_whatever_the_first_chose() -> None:
    domain = create_prisoners_dilemma_domain(ONE_ROUND)

    assert create_agent(200, seed=3).search(domain, after("cooperate")) == create_agent(200, seed=3).search(domain, after("defect"))


def test_the_second_player_s_semi_determinized_search_weighs_the_first_s_two_choices_and_still_defects(
    caplog: pytest.LogCaptureFixture,
) -> None:
    domain = create_prisoners_dilemma_domain(ONE_ROUND)
    agent = AgentBuilder().with_iterations(300).with_exploration(EXPLORATION).with_seed(1).with_theory_of_mind().build()

    result = agent.search(domain, after("cooperate"))

    assert result.chosen == choose("defect")
    assert sorted((hypothesis.label, hypothesis.probability) for hypothesis in result.hypotheses) == [
        ((("chosen(A)", "cooperate"),), 0.5),
        ((("chosen(A)", "defect"),), 0.5),
    ]
    assert "B weighs 2 hypotheses: chosen(A)='cooperate' at 0.5; chosen(A)='defect' at 0.5; 150, 150 iterations" in caplog.messages


SIMULTANEOUS_ONE_ROUND = replace(STANDARD, name="simultaneousoneround", rounds=1, simultaneous=True)


def test_a_simultaneous_round_plays_both_choices_at_once() -> None:
    domain = create_prisoners_dilemma_domain(SIMULTANEOUS_ONE_ROUND)
    joint = JointAction((("A", choose("cooperate")), ("B", choose("defect"))))

    ((state, _),) = create_predictor().predict_joint(domain.transitions, domain.initial_state, joint).outcomes

    variables = dict(state.variables)
    assert (variables["payoff(A)"], variables["payoff(B)"], variables["turn(A)"], variables["turn(B)"]) == (0, 5, False, False)


def test_in_one_simultaneous_round_both_players_mostly_defect() -> None:
    domain = create_prisoners_dilemma_domain(SIMULTANEOUS_ONE_ROUND)
    agent = AgentBuilder().with_iterations(2000).with_exploration(EXPLORATION).with_seed(1).build()

    for player in ("A", "B"):
        assert dict(agent.search(domain, domain.initial_state, player).strategy)[choose("defect")] > 0.7


def test_exact_search_refuses_the_hidden_choices() -> None:
    domain = create_prisoners_dilemma_domain(ONE_ROUND)

    with pytest.raises(ValueError, match="hidden information"):
        ExactSearch(create_solver(), create_predictor(), StateReader()).value(domain, domain.initial_state)
