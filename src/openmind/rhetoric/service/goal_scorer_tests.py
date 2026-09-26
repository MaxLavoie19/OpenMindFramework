import pytest

from openmind.rhetoric.model.distance import Distance
from openmind.rhetoric.model.distance_target import DistanceTarget
from openmind.rhetoric.model.rhetorical_goal import RhetoricalGoal
from openmind.rhetoric.service.goal_scorer import GoalScorer

IDENTITY = "identity"


def a_target(**held) -> DistanceTarget:
    fields = {
        "member": "a student",
        "question": "whether it is so",
        "kind": IDENTITY,
        "value": 0.0,
        "problematicity": None,
    }
    return DistanceTarget(**{**fields, **held})


def a_distance(**held) -> Distance:
    fields = {
        "member": "a student",
        "question": "whether it is so",
        "kind": IDENTITY,
        "value": 0.0,
        "problematicity": 0.0,
    }
    return Distance(**{**fields, **held})


def test_a_goal_wanting_nothing_is_met():
    """A speaker with no targets has nothing to fall short of, and a planner needs a payoff rather than a
    division by nothing."""
    assert GoalScorer().score(RhetoricalGoal(()), []) == 1.0


def test_a_target_hit_exactly_scores_the_whole_of_it():
    goal = RhetoricalGoal((a_target(value=1.0),))

    assert GoalScorer().score(goal, [a_distance(value=1.0)]) == pytest.approx(1.0)


def test_missing_by_the_widest_a_distance_can_be_scores_nothing():
    """Two is the widest gap there is, so a miss of two is a complete miss rather than a number that keeps
    growing and drives the score below nothing."""
    goal = RhetoricalGoal((a_target(value=-1.0),))

    assert GoalScorer().score(goal, [a_distance(value=1.0)]) == pytest.approx(0.0)


def test_missing_by_more_than_the_widest_still_scores_nothing_rather_than_less():
    """A planner's payoff has a floor: a goal missed twice over is not worse to a planner than one missed
    once, and a negative payoff would be read as worse than losing."""
    goal = RhetoricalGoal((a_target(value=-2.0),))

    assert GoalScorer().score(goal, [a_distance(value=2.0)]) == pytest.approx(0.0)


def test_a_target_whose_distance_was_never_measured_misses_fully():
    """Not having asked is not the same as being close, and a speaker scored well for what it never measured
    would learn to measure nothing."""
    goal = RhetoricalGoal((a_target(value=0.0),))

    assert GoalScorer().score(goal, []) == pytest.approx(0.0)


def test_a_distance_about_somebody_else_does_not_answer_this_target():
    """Targets are per member, per question and per kind — a bystander's agreement is not the wrongdoer's."""
    goal = RhetoricalGoal((a_target(member="a student", value=0.0),))

    assert GoalScorer().score(goal, [a_distance(member="a bystander", value=0.0)]) == pytest.approx(0.0)


def test_how_much_the_gap_matters_is_scored_beside_how_wide_it_is():
    """A goal can want a gap and want it to matter — confronting a wrongdoer targets a large distance *and*
    raised problematicity — so the two are scored together rather than one standing for both."""
    goal = RhetoricalGoal((a_target(value=1.0, problematicity=1.0),))
    scorer = GoalScorer()

    both = scorer.score(goal, [a_distance(value=1.0, problematicity=1.0)])
    one = scorer.score(goal, [a_distance(value=1.0, problematicity=-1.0)])

    assert both == pytest.approx(1.0)
    assert one == pytest.approx(0.5)


def test_a_target_wanting_neither_a_value_nor_a_problematicity_asks_nothing():
    """A target with nothing set says nothing about what would be good, so it neither helps nor hurts the
    score — and it must not divide it by a weight it never earned."""
    goal = RhetoricalGoal((a_target(value=None), a_target(member="another", value=0.0)))

    assert GoalScorer().score(goal, [a_distance(member="another", value=0.0)]) == pytest.approx(1.0)


def test_a_heavier_target_moves_the_score_more():
    """What a speaker wants most should count most, which is the whole of what a weight is for."""
    light = RhetoricalGoal((a_target(value=0.0, weight=1.0), a_target(member="another", value=0.0, weight=1.0)))
    heavy = RhetoricalGoal((a_target(value=0.0, weight=9.0), a_target(member="another", value=0.0, weight=1.0)))
    missed = [a_distance(value=2.0), a_distance(member="another", value=0.0)]
    scorer = GoalScorer()

    assert scorer.score(heavy, missed) < scorer.score(light, missed)
