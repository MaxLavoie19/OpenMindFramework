from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DistanceTarget:
    """The distance and problematicity a speaker wants on a question, for one audience member and one kind, and how much
    reaching it weighs in the goal. A target left None doesn't count."""

    member: str
    question: str
    kind: str
    value: float | None
    problematicity: float | None
    weight: float = 1.0
