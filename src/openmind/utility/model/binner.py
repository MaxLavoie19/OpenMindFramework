from typing import Protocol

from openmind.predictor.model.outcome_distribution import OutcomeDistribution
from openmind.utility.model.bin import Bin


class Binner[Model](Protocol):
    """The binning task: a distribution of outcomes grouped into bins, each with its likelihood, so that continuous
    values can be weighed. Models bin evenly, around a decision threshold, by rules, or any other way."""

    def bins(self, model: Model, outcomes: OutcomeDistribution, player: str, payoff: str) -> tuple[Bin, ...]: ...
