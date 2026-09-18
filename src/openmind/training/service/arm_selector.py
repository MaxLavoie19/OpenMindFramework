import logging
import math
import random
from collections.abc import Mapping, Sequence

logger = logging.getLogger(__name__)


class ArmSelector:
    """Chooses which two arms, the models agents follow, play a game against each other, by UCB1 on their game scores. An
    arm's score is its points per game, a win 1, a draw 0.5, a loss 0. Games under way count as games without points yet,
    so arms playing now aren't chosen over and over before their results come back; an arm with games under way and none
    finished scores 0.5 meanwhile. An arm never chosen goes first."""

    def pair(
        self,
        scores: Mapping[str, tuple[int, float]],
        pending: Mapping[str, int],
        arms: Sequence[str],
        exploration: float,
        rng: random.Random,
    ) -> tuple[str, str]:
        """The two arms with the highest upper confidence bounds, the highest first; ties are drawn at random. `scores`
        gives each arm's finished games and points, `pending` its games under way; an arm missing from either has none.
        Fewer than two arms raise ValueError."""
        if len(set(arms)) < 2:
            raise ValueError(f"A game between arms needs two arms, not {len(set(arms))}")
        counts = {arm: scores.get(arm, (0, 0.0))[0] + pending.get(arm, 0) for arm in arms}
        total = sum(counts.values())

        def bound(arm: str) -> float:
            if counts[arm] == 0:
                return math.inf
            games, points = scores.get(arm, (0, 0.0))
            mean = points / games if games else 0.5
            return mean + exploration * math.sqrt(math.log(max(total, 1)) / counts[arm])

        ranked = sorted(dict.fromkeys(arms), key=lambda arm: (-bound(arm), rng.random()))
        return ranked[0], ranked[1]
