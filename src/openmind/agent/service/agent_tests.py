import math

import pytest

from openmind.agent.factory.rock_paper_scissors_factory import create_rock_paper_scissors_domain
from openmind.agent.model.domain import Domain
from openmind.agent.service.agent import Agent
from openmind.csp.factory.csp_factory import create_solver
from openmind.csp.model.action_definition import ActionDefinition
from openmind.csp.model.problem import Problem
from openmind.mcts.model.search_settings import SearchSettings
from openmind.mcts.service.semi_determinized_search import SemiDeterminizedSearch
from openmind.mcts.service.semi_determinized_search_tests import Believes, coin_domain
from openmind.mcts.service.tree_search import TreeSearch
from openmind.mcts.service.tree_search_tests import Ticking
from openmind.observation.factory.state_observer_factory import create_state_observer
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule
from openmind.timing.model.clock import Clock
from openmind.timing.service.plain_time_budget_estimator import PlainTimeBudgetEstimator
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.model.action import Action
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

pytestmark = pytest.mark.log_level("INFO")


def win_or_lose() -> Domain:
    no_payoff = PythonRule("payoff is None")
    return Domain(
        "win or lose",
        State((("payoff", None), ("turn", "me"))),
        Problem((ActionDefinition("lose", (), (no_payoff,)), ActionDefinition("win", (), (no_payoff,)))),
        TransitionModel(
            (
                Transition("lose", (Branch(1.0, PythonRule("payoff = 0.0")),)),
                Transition("win", (Branch(1.0, PythonRule("payoff = 1.0")),)),
            )
        ),
        Players(("me",), "turn", ("payoff",)),
    )


def new_agent() -> Agent:
    tree_search = TreeSearch(create_solver(), create_predictor(), StateReader(), ActionTextMapper())
    return Agent(tree_search, SearchSettings(50, math.sqrt(2), 1))


def test_choose_returns_the_most_visited_action() -> None:
    domain = win_or_lose()

    assert new_agent().choose(domain, domain.initial_state) == Action("win", ())


def test_search_returns_the_statistics_and_the_tree_samples() -> None:
    domain = win_or_lose()

    result = new_agent().search(domain, domain.initial_state)

    assert result.chosen == Action("win", ())
    assert {sample.action.name for sample in result.samples} == {"lose", "win"}


def timed_agent(iterations: int | None, estimator: bool = True, theory: bool = False) -> Agent:
    """An agent whose search reads a time source moving on by a second a reading, estimating a step's budget as all the
    time left."""
    tree_search = TreeSearch(create_solver(), create_predictor(), StateReader(), ActionTextMapper(), time_source=Ticking())
    semi_determinized = (
        SemiDeterminizedSearch(tree_search, create_state_observer(), StateReader(), ActionTextMapper()) if theory else None
    )
    return Agent(
        tree_search,
        SearchSettings(iterations, math.sqrt(2), 1),
        semi_determinized_search=semi_determinized,
        theory=Believes(0.5) if theory else None,
        estimator=PlainTimeBudgetEstimator(1) if estimator else None,
    )


def test_an_agent_given_a_clock_searches_for_the_step_s_budget() -> None:
    domain = win_or_lose()

    result = timed_agent(None).search(domain, domain.initial_state, clock=Clock(5.0), steps_played=3)

    assert result.iterations == 5
    assert result.chosen == Action("win", ())


def test_an_agent_s_iterations_cap_its_search_on_a_clock() -> None:
    domain = win_or_lose()

    assert timed_agent(3).search(domain, domain.initial_state, clock=Clock(5.0)).iterations == 3


def test_an_agent_without_a_clock_searches_its_iterations() -> None:
    domain = win_or_lose()

    assert timed_agent(50).search(domain, domain.initial_state).iterations == 50


def test_an_agent_given_a_clock_without_an_estimator_raises() -> None:
    domain = win_or_lose()

    with pytest.raises(ValueError, match="time budget estimator"):
        timed_agent(50, estimator=False).choose(domain, domain.initial_state, clock=Clock(5.0))


def test_an_agent_built_without_iterations_needs_a_clock() -> None:
    domain = win_or_lose()

    with pytest.raises(ValueError, match="needs a clock"):
        timed_agent(None).choose(domain, domain.initial_state)


def test_a_semi_determinized_agent_shares_the_step_s_budget_between_hypotheses() -> None:
    domain = coin_domain("tails", 0.5)

    result = timed_agent(None, theory=True).search(domain, domain.initial_state, clock=Clock(10.0))

    assert len(result.hypotheses) == 2
    assert result.iterations == 6


def test_an_agent_acting_at_once_searches_for_the_step_s_budget() -> None:
    domain = create_rock_paper_scissors_domain()

    assert timed_agent(None).search(domain, domain.initial_state, "A", Clock(5.0)).iterations == 5
