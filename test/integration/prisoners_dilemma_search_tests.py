from dataclasses import replace

import pytest

from openmind.agent.constant.prisoners_dilemma_constant import STANDARD
from openmind.agent.factory.agent_factory import create_agent
from openmind.agent.factory.prisoners_dilemma_factory import create_prisoners_dilemma_domain
from openmind.csp.factory.csp_factory import create_solver
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.world.model.action import Action
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


def test_exact_search_refuses_the_hidden_choices() -> None:
    domain = create_prisoners_dilemma_domain(ONE_ROUND)

    with pytest.raises(ValueError, match="hidden information"):
        ExactSearch(create_solver(), create_predictor(), StateReader()).value(domain, domain.initial_state)
