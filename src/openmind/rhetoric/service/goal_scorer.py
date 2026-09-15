import math
from collections.abc import Sequence

from openmind.rhetoric.model.distance import Distance
from openmind.rhetoric.model.rhetorical_goal import RhetoricalGoal


class GoalScorer:
    """Scores how close distances are to a rhetorical goal, from 0 to 1, a planner's payoff: 1 minus the weighted mean
    miss, where a target's miss is the gap between the distance and its target value (out of 2, the widest a distance
    can be) and between the problematicity and its target (out of 2), averaged over what the target sets. A target whose
    distance wasn't measured misses fully. A goal without targets scores 1."""

    def score(self, goal: RhetoricalGoal, distances: Sequence[Distance]) -> float:
        measured = {(distance.member, distance.question, distance.kind): distance for distance in distances}
        weights = misses = 0.0
        for target in goal.targets:
            parts: list[float] = []
            distance = measured.get((target.member, target.question, target.kind))
            if target.value is not None:
                parts.append(1.0 if distance is None else min(1.0, abs(distance.value - target.value) / 2.0))
            if target.problematicity is not None:
                parts.append(1.0 if distance is None else min(1.0, abs(distance.problematicity - target.problematicity) / 2.0))
            if parts:
                weights += target.weight
                misses += target.weight * math.fsum(parts) / len(parts)
        return 1.0 if weights == 0.0 else 1.0 - misses / weights
