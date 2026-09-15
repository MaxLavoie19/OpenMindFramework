from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueSettings:
    """How value rules are generated and fitted: the L1 prices swept; when a fit stops, after max_steps or once no weight
    moves by more than tolerance; and the expression search's budget: how many seconds it runs, how many bytes its
    process holds, and how many candidates it tries, None for no limit."""

    prices: tuple[float, ...]
    max_steps: int
    tolerance: float
    seconds: float
    memory_bytes: int
    candidates: int | None = None
