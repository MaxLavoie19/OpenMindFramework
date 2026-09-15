from dataclasses import dataclass

from openmind.inference.model.deduction_budget import DeductionBudget


@dataclass(frozen=True, slots=True)
class PonderingSettings:
    """How many positions a round ponders, those its previous rules missed most, and the budget each deduction gets."""

    positions: int
    budget: DeductionBudget
