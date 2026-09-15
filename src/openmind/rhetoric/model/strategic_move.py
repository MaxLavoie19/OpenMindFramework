from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategicMove:
    """What a planner decides, before any wording: increase or decrease the distance, or the problematicity of the
    distance, on a question, of one kind, with one audience member. A sub-skill turns it into a concrete message."""

    member: str
    question: str
    kind: str
    aspect: str
    direction: str
