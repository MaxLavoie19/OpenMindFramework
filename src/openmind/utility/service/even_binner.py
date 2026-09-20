import logging

from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.structure.model.map import Map
from openmind.utility.model.bin import Bin

logger = logging.getLogger(__name__)

#: How many bins an even binning makes when it isn't told otherwise.
BINS = 3


class EvenBinner:
    """The binning model that cuts the range of payoffs into bins of equal width: enough for most decisions, and what
    binning falls back on before a model is fitted for it.

    Each bin holds the outcomes whose payoff falls in it, and its likelihood is their probabilities summed. Outcomes
    without a payoff for the player are left out; a range of no width gives one bin."""

    def __init__(self, bins: int = BINS) -> None:
        self._bins = bins

    def bins(self, model: object, outcomes: OutcomeDistribution, player: str, payoff: str) -> tuple[Bin, ...]:
        """The bins the outcomes fall in, lowest first; empty where none carries a payoff for the player."""
        paid = [
            (value, probability)
            for outcome, probability in outcomes.outcomes
            if (value := self._payoff(outcome, player, payoff)) is not None
        ]
        if not paid:
            return ()
        low, high = min(value for value, _ in paid), max(value for value, _ in paid)
        if low == high:
            return (Bin(low, high, sum(probability for _, probability in paid)),)
        width = (high - low) / self._bins
        edges = [(low + width * index, low + width * (index + 1)) for index in range(self._bins)]
        binned = tuple(
            Bin(start, end, sum(probability for value, probability in paid if start <= value < end or (index == self._bins - 1 and value == high)))
            for index, (start, end) in enumerate(edges)
        )
        logger.debug("Binned %d outcomes from %g to %g into %d bins", len(paid), low, high, self._bins)
        return binned

    def _payoff(self, outcome: object, player: str, payoff: str) -> float | None:
        held = outcome.model(payoff) if outcome.has(payoff) else None  # type: ignore[attr-defined]
        if not isinstance(held, Map):
            return None
        value = held.get(player)
        if isinstance(value, bool) or not isinstance(value, int | float):
            return None
        return float(value)
