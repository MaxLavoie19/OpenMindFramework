from dataclasses import dataclass

from openmind.rbs.model.rule import Rule


@dataclass(frozen=True, slots=True)
class RemovalTest:
    """One try at removing a rule: the pass it was tried in; without it, the mean rise in regret over the full set, the
    upper bound of that rise and the change in optimal choices; whether it was removed, whether without a search
    because it decided no rating, and the seconds the try took."""

    rule: Rule
    pass_number: int
    regret_difference: float
    upper_bound: float
    optimal_difference: int
    removed: bool
    free: bool
    seconds: float
