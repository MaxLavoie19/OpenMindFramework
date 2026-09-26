from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueSettings:
    """How value rules are generated and fitted: the L1 prices swept; when a fit stops, after max_steps or once no weight
    moves by more than tolerance; and the expression search's budget: how many seconds it runs, how many bytes its
    process holds, and how many candidates it tries, None for no limit.

    `keep_every_price` declares each price's fit as a model of its own rather than only the one the held-out rows
    chose. The sweep fits them all and throws all but one away, and they are not small differences — a sparse fit
    of four terms and a dense one of sixty-eight are different heuristics, not one heuristic at two settings.
    Which of them is worth playing with is a question no loss answers, so where something means to find out by
    playing, they are kept for it to play."""

    prices: tuple[float, ...]
    max_steps: int
    tolerance: float
    seconds: float
    memory_bytes: int
    candidates: int | None = None
    keep_every_price: bool = False
