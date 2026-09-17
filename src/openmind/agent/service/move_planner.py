import logging
import math

from openmind.mcts.constant.mcts_constant import FULL_OPTION, ONE_PLY_OPTION, RANDOM_OPTION, SEARCH_OPTION

logger = logging.getLogger(__name__)


class MovePlanner:
    """Picks how a player on a clock chooses its move, the richest option whose cost fits the move's budget, from what
    each option has cost so far in the game:

    - the fallback then the search needs the fallback's cost per legal move and one search iteration per legal move;
    - the search alone needs one search iteration per legal move, since a search with fewer iterations than moves
      hasn't looked at each move once;
    - one-ply needs one valuation per legal move;
    - a random move needs nothing, and is what a budget of 0 or less gets.

    A cost never measured yet counts as fitting, so the first move of a game tries the richest option and its deadline
    cuts it short; each option's cost is the mean of what it cost in this game."""

    def __init__(self) -> None:
        self._costs: dict[str, list[float]] = {}

    def plan(self, budget: float, moves: int, has_fallback: bool, has_valuer: bool) -> str:
        if budget <= 0.0 or moves <= 0:
            return RANDOM_OPTION
        iteration, fallback, valuation = self._mean("iteration"), self._mean("fallback"), self._mean("valuation")
        search = None if iteration is None else iteration * moves
        if has_fallback and has_valuer and self._fits(budget, search, None if fallback is None else fallback * moves):
            return FULL_OPTION
        if self._fits(budget, search):
            return SEARCH_OPTION
        per_move = valuation if valuation is not None else fallback
        if has_valuer and self._fits(budget, None if per_move is None else per_move * moves):
            return ONE_PLY_OPTION
        return RANDOM_OPTION

    def observe(self, kind: str, seconds: float, count: int) -> None:
        """What `count` of a kind of step cost together: `iteration` for search iterations, `fallback` for the fallback's
        valuing of `count` legal moves, `valuation` for one-ply's."""
        if count > 0 and math.isfinite(seconds):
            self._costs.setdefault(kind, []).append(seconds / count)

    def _mean(self, kind: str) -> float | None:
        costs = self._costs.get(kind)
        return None if not costs else math.fsum(costs) / len(costs)

    def _fits(self, budget: float, *costs: float | None) -> bool:
        return math.fsum(cost for cost in costs if cost is not None) <= budget
