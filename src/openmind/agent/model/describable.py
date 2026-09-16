from typing import Protocol


class Describable(Protocol):
    """A part of an agent that can say what it is, word for word: enough, as JSON text, to build it again."""

    def describe(self) -> str: ...
