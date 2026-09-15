"""Scenarios with an agreed answer: exact search over the first, hand-written reaction rules must pick the strategic move
a person would call right. The expected values were worked out by hand from those rules (a step of 0.5, persuasion or
accommodation at even chances), so these tests check that the planner follows the rules, not that the rules describe
people."""

import pytest

from openmind.csp.factory.csp_factory import create_solver
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rhetoric.factory.rhetoric_factory import create_rhetoric_domain
from openmind.rhetoric.factory.rhetoric_factory_tests import football
from openmind.rhetoric.model.distance_target import DistanceTarget
from openmind.rhetoric.model.ethos import Ethos
from openmind.rhetoric.model.pathos import Pathos
from openmind.rhetoric.model.position import Position
from openmind.rhetoric.model.rhetorical_goal import RhetoricalGoal
from openmind.rhetoric.model.rhetorical_scenario import RhetoricalScenario
from openmind.rhetoric.model.speaker import Speaker
from openmind.world.model.action import Action
from openmind.world.service.state_reader import StateReader


def best(scenario: RhetoricalScenario) -> tuple[set[tuple[str, str, str, str]], float]:
    """The optimal first moves as (member, kind, aspect, direction), and their value."""
    domain, search = create_rhetoric_domain(scenario), ExactSearch(create_solver(), create_predictor(), StateReader())
    values = dict(search.action_values(domain, domain.initial_state))
    optimal = search.optimal_actions(domain, domain.initial_state)
    return {_move(action) for action in optimal}, values[optimal[0]]


def _move(action: Action) -> tuple[str, str, str, str]:
    parameters = dict(action.parameters)
    return (str(parameters["member"]), str(parameters["kind"]), str(parameters["aspect"]), str(parameters["direction"]))


def test_to_calm_a_fan_it_is_best_to_make_the_team_matter_less_than_to_argue_about_it() -> None:
    moves, value = best(football())

    assert moves == {("joe", "audience", "problematicity", "decrease")}
    assert value == pytest.approx(0.3)


def test_to_confront_a_cheater_before_a_bystander_it_raises_the_stakes_with_him_and_wins_over_the_bystander() -> None:
    cheated = "joe cheated"
    speaker = Speaker(
        "ann",
        Ethos((Position(cheated, 1.0, 1.0),)),
        (("joe", Ethos((Position(cheated, 1.0),))), ("bob", Ethos((Position(cheated, 1.0),)))),
        (("joe", Pathos((Position(cheated, -1.0, 0.3),))), ("bob", Pathos((Position(cheated, 0.0, 0.3),)))),
    )
    goal = RhetoricalGoal(
        (
            DistanceTarget("joe", cheated, "audience", 2.0, 2.0),
            DistanceTarget("bob", cheated, "audience", 0.0, 0.0),
        )
    )

    moves, value = best(RhetoricalScenario("cheating", speaker, goal, 2))

    assert moves == {("joe", "audience", "problematicity", "increase"), ("bob", "audience", "distance", "decrease")}
    assert value == pytest.approx(0.8565625)


def test_a_teacher_reassures_a_nervous_student_rather_than_pretend_to_know_less() -> None:
    chess = "knows chess"
    speaker = Speaker(
        "teacher",
        Ethos((Position(chess, 0.9, 0.3),)),
        (("student", Ethos((Position(chess, 0.9),))),),
        (("student", Pathos((Position(chess, -0.5, 0.9),))),),
    )
    goal = RhetoricalGoal((DistanceTarget("student", chess, "audience", 1.0, 0.2),))

    moves, value = best(RhetoricalScenario("lesson", speaker, goal, 1))

    assert moves == {("student", "audience", "problematicity", "decrease")}
    assert value == pytest.approx(0.81)
