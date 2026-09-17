import logging
import math

from openmind.mcts.constant.mcts_constant import FULL_OPTION, RANDOM_OPTION, SEARCH_OPTION

logger = logging.getLogger(__name__)


class MovePlanner:
    """Picks how a player on a clock chooses its move, from what each way has cost so far in the game:

    - the fallback then the search, when the agent has both and the fallback's cost, per legal move, leaves time to
      search; a fallback never measured yet counts as leaving time, and its deadline cuts it short;
    - the search alone otherwise, however few iterations the budget pays for: exploring, even at random, beats a move
      chosen without searching;
    - a random move only on a budget of 0, as a player at or below its reserve gets.

    A cost is the mean of what it cost in this game."""

    def __init__(self) -> None:
        self._costs: dict[str, list[float]] = {}

    def plan(self, budget: float, moves: int, has_fallback: bool, has_valuer: bool) -> str:
        if budget <= 0.0 or moves <= 0:
            return RANDOM_OPTION
        fallback = self._mean("fallback")
        if has_fallback and has_valuer and (fallback is None or fallback * moves < budget):
            return FULL_OPTION
        return SEARCH_OPTION

    def observe(self, kind: str, seconds: float, count: int) -> None:
        """What `count` of a kind of step cost together: `fallback` for the fallback on `count` legal moves, `iteration`
        for search iterations."""
        if count > 0 and math.isfinite(seconds):
            self._costs.setdefault(kind, []).append(seconds / count)

    def _mean(self, kind: str) -> float | None:
        costs = self._costs.get(kind)
        return None if not costs else math.fsum(costs) / len(costs)
