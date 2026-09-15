from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Distance:
    """How far apart two positions on a question are, for one audience member and one kind (identity or audience):
    signed, the first position's answer minus the second's, so a teacher knowing more than a student is positive; and
    its problematicity, how much the gap matters: its size times the importance of the question to whoever holds the
    second position."""

    member: str
    question: str
    kind: str
    value: float
    problematicity: float
