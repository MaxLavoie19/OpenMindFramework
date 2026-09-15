from dataclasses import dataclass

from openmind.inference.model.deduction_budget import DeductionBudget


@dataclass(frozen=True, slots=True)
class PonderingSettings:
    """How many positions a round ponders, those its previous rules missed most, the budget each deduction gets, and how
    many positions of decisive games it deduces at most walking back from their ends, 0 walking none."""

    positions: int
    budget: DeductionBudget
    endings: int = 0
