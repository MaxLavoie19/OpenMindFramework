import pytest

from openmind.rhetoric.model.distance import Distance
from openmind.rhetoric.model.distance_target import DistanceTarget
from openmind.rhetoric.model.rhetorical_goal import RhetoricalGoal
from openmind.rhetoric.service.goal_scorer import GoalScorer

CHEATED = "joe cheated"
DISTANCES = (
    Distance("joe", CHEATED, "audience", 2.0, 2.0),
    Distance("ann", CHEATED, "audience", 1.0, 0.5),
)


def test_de_escalating_and_confronting_score_the_same_distances_in_opposite_ways() -> None:
    de_escalate = RhetoricalGoal((DistanceTarget("joe", CHEATED, "audience", 0.0, 0.0),))
    confront = RhetoricalGoal((DistanceTarget("joe", CHEATED, "audience", 2.0, 2.0),))

    assert GoalScorer().score(de_escalate, DISTANCES) == 0.0
    assert GoalScorer().score(confront, DISTANCES) == 1.0


def test_targets_are_weighted_and_a_missing_distance_misses_fully() -> None:
    goal = RhetoricalGoal(
        (
            DistanceTarget("joe", CHEATED, "audience", 2.0, None, weight=1.0),
            DistanceTarget("ann", CHEATED, "audience", 0.0, 0.0, weight=3.0),
            DistanceTarget("bob", CHEATED, "audience", 0.0, None, weight=0.0),
        )
    )

    assert GoalScorer().score(goal, DISTANCES) == pytest.approx(1.0 - (1.0 * 0.0 + 3.0 * (0.5 + 0.25) / 2) / 4.0)
    assert GoalScorer().score(RhetoricalGoal((DistanceTarget("bob", CHEATED, "audience", 0.0, None),)), DISTANCES) == 0.0
    assert GoalScorer().score(RhetoricalGoal(()), DISTANCES) == 1.0
