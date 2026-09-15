from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DeductionBudget:
    """How far and how long one position is reasoned about: at most `plies` actions ahead, within `seconds`; `highest` is
    the highest payoff a player can get, so a move proven to reach it needs no comparison with moves not proven yet."""

    plies: int
    seconds: float
    highest: float = 1.0
