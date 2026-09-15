from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A body read at the indices of a base, at every index `i` or at every pair of different indices `i` and `j`, then
    summed, counted where it holds, or taken at its lowest or highest; the body is Python source reading its position
    where VIEW stands, and has its own clauses and how many actions it looks ahead.

    `readings` and `operations` are the body's own parts, in the order they were added: the first reading alone, then
    each operation with the reading it brought, so `body` is `fold(operations, readings)`. A body built another way, such
    as one comparing a body with itself at `j`, has no parts and is evaluated from its source."""

    base: str
    pair: bool
    kind: str
    body: str
    body_clauses: int
    body_plies: int = 0
    readings: tuple[str, ...] = ()
    operations: tuple[str, ...] = ()
