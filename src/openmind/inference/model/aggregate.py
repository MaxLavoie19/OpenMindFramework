from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A body read at the indices of a base, at every index `i` or at every pair of different indices `i` and `j`, then
    summed, counted where it holds, or taken at its lowest or highest; the body is Python source reading its position
    where VIEW stands, and has its own clauses."""

    base: str
    pair: bool
    kind: str
    body: str
    body_clauses: int
