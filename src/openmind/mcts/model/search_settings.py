from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """How long and how widely to search: iterations, UCT exploration weight, and random seed (None is unseeded)."""

    iterations: int
    exploration: float
    seed: int | None
