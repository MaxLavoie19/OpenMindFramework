from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PonderingSummary:
    """A round's pondering in numbers: the positions pondered, those proven, the seeds induced, the seeds the expression
    search kept, and the seeds among the value rules."""

    positions: int
    proven: int
    seeds: int
    seeds_kept: int
    seeds_in_rules: int
