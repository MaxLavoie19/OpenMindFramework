from typing import Protocol

from openmind.language.model.meaning import Meaning


class TextEncoder(Protocol):
    """Turns a meaning into English. Everything the sentence says is in the meaning: an encoder adds no knowledge of
    its own. Templates, a trained model or a language model can implement it."""

    @property
    def name(self) -> str: ...

    def encode(self, meaning: Meaning) -> str | None: ...
