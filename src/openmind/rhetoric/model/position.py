from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position:
    """An answer to a question, in Michel Meyer's sense: "is his fish disgusting" answered from -1 (no) to 1 (yes). Its
    importance, from 0 to 1, is how much the question matters to whoever holds the position; it stays with them, so it
    is None when it isn't known, such as in a message heard."""

    question: str
    answer: float
    importance: float | None = None
