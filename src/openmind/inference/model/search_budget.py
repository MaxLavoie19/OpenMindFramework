from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """How long an expression search runs at most, in seconds; how many bytes its process holds at most, workers each
    holding an even share; and how many candidates it tries at most, None for no limit."""

    seconds: float
    memory_bytes: int
    candidates: int | None = None
