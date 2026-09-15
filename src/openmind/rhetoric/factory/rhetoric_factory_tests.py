from dataclasses import replace

import pytest

from openmind.csp.factory.csp_factory import create_solver
from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.rhetoric.factory.rhetoric_factory import create_rhetoric_domain
from openmind.rhetoric.model.distance_target import DistanceTarget
from openmind.rhetoric.model.ethos import Ethos
from openmind.rhetoric.model.pathos import Pathos
from openmind.rhetoric.model.position import Position
from openmind.rhetoric.model.rhetorical_goal import RhetoricalGoal
from openmind.rhetoric.model.rhetorical_scenario import RhetoricalScenario
from openmind.rhetoric.model.speaker import Speaker
from openmind.rhetoric.service.distance_measurer import DistanceMeasurer
from openmind.rhetoric.service.goal_scorer import GoalScorer
from openmind.world.model.action import Action

TEAM = "joe's team is good"


def football(moves: int = 1) -> RhetoricalScenario:
    """Ann told Joe his team is bad; Joe, she perceives, loves his team. She wants to calm things down."""
    speaker = Speaker(
        "ann",
        Ethos((Position(TEAM, -1.0, 0.2),)),
        (("joe", Ethos((Position(TEAM, -1.0),))),),
        (("joe", Pathos((Position(TEAM, 1.0, 0.9),))),),
    )
    goal = RhetoricalGoal((DistanceTarget("joe", TEAM, "audience", 0.0, 0.0),))
    return RhetoricalScenario("football", speaker, goal, moves)


def move(kind: str, aspect: str, direction: str, member: str = "joe") -> Action:
    return Action("move", (("aspect", aspect), ("direction", direction), ("kind", kind), ("member", member), ("question", "q1")))


def test_the_state_holds_what_the_speaker_knows_and_perceives() -> None:
    variables = dict(create_rhetoric_domain(football(2)).initial_state.variables)

    assert variables == {
        "effective_answer(q1)": -1.0,
        "effective_importance(q1)": 0.2,
        "shown_answer(joe,q1)": -1.0,
        "perceived_answer(joe,q1)": 1.0,
        "perceived_importance(joe,q1)": 0.9,
        "moves_left": 2,
        "turn": "ann",
        "payoff": None,
    }


def test_every_strategic_move_is_legal_until_the_exchange_is_scored() -> None:
    domain, solver, predictor = create_rhetoric_domain(football()), create_solver(), create_predictor()

    actions = solver.solve(domain.problem, domain.initial_state)
    ((scored, _), _) = predictor.predict(domain.transitions, domain.initial_state, actions[0]).outcomes

    assert len(actions) == 8
    assert solver.solve(domain.problem, scored) == ()


def test_decreasing_the_audience_distance_persuades_or_accommodates() -> None:
    domain, predictor = create_rhetoric_domain(football(2)), create_predictor()

    (persuaded, first), (accommodated, second) = predictor.predict(
        domain.transitions, domain.initial_state, move("audience", "distance", "decrease")
    ).outcomes

    assert (first, second) == (0.5, 0.5)
    assert dict(persuaded.variables)["perceived_answer(joe,q1)"] == pytest.approx(0.95)
    assert dict(persuaded.variables)["shown_answer(joe,q1)"] == -1.0
    assert dict(accommodated.variables)["shown_answer(joe,q1)"] == pytest.approx(-0.5)
    assert dict(accommodated.variables)["moves_left"] == 1


@pytest.mark.parametrize(
    ("chosen", "variable", "value"),
    [
        (move("audience", "problematicity", "decrease"), "perceived_importance(joe,q1)", 0.4),
        (move("audience", "problematicity", "increase"), "perceived_importance(joe,q1)", 1.0),
        (move("audience", "distance", "increase"), "shown_answer(joe,q1)", -1.0),
        (move("identity", "distance", "increase"), "shown_answer(joe,q1)", -0.5),
        (move("identity", "problematicity", "decrease"), "effective_importance(q1)", 0.0),
    ],
)
def test_the_other_moves_shift_answers_and_importances_by_a_step(chosen: Action, variable: str, value: float) -> None:
    domain = create_rhetoric_domain(football(2))

    outcomes = create_predictor().predict(domain.transitions, domain.initial_state, chosen).outcomes

    assert all(dict(outcome.variables)[variable] == pytest.approx(value) for outcome, _ in outcomes)


def test_the_last_move_pays_the_goal_score_of_the_distances_it_leaves() -> None:
    scenario = football()
    domain = create_rhetoric_domain(scenario)

    ((outcome, _), _) = create_predictor().predict(
        domain.transitions, domain.initial_state, move("audience", "problematicity", "decrease")
    ).outcomes

    speaker = replace(scenario.speaker, projective_pathos=(("joe", Pathos((Position(TEAM, 1.0, 0.4),))),))
    expected = GoalScorer().score(scenario.goal, DistanceMeasurer().distances(speaker))
    assert dict(outcome.variables)["payoff"] == pytest.approx(expected) == pytest.approx(0.3)


def test_names_and_questions_the_state_can_t_hold_are_refused() -> None:
    scenario = football()
    comma = replace(scenario, speaker=replace(scenario.speaker, projective_ethos=(("joe, jr", Ethos((Position(TEAM, -1.0),))),)))
    unknown = replace(scenario, goal=RhetoricalGoal((DistanceTarget("joe", "is fish good", "audience", 0.0, None),)))

    with pytest.raises(ValueError, match="name"):
        create_rhetoric_domain(comma)
    with pytest.raises(ValueError, match="doesn't answer"):
        create_rhetoric_domain(unknown)
