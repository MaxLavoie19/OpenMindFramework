from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchSettings:
    """How much a planner may explore. `nodes` is how many nodes it may expand, None for as many as it needs; `depth`
    how far it may look, None for as far as the game goes; `seed` what its random choices follow.

    How much to spend is the time management policy's at the budget step; until then a caller says."""

    nodes: int | None = None
    depth: int | None = None
    seed: int = 0
