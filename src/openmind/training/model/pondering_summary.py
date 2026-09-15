from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PonderingSummary:
    """A round's pondering in numbers: the positions pondered, those proven, the seeds induced, the seeds the expression
    search kept, the seeds among the value rules, and the positions of decisive games deduced walking back from their
    ends, and those proven."""

    positions: int
    proven: int
    seeds: int
    seeds_kept: int
    seeds_in_rules: int
    endings_deduced: int = 0
    endings_proven: int = 0
