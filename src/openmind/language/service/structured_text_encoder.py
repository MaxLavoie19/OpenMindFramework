from openmind.language.constant.structured_language_constant import (
    ANSWER_FIELD,
    QUESTION_FIELD,
    SPECIAL_CHARACTERS,
    STRUCTURED_LANGUAGE,
)
from openmind.rhetoric.constant.rhetoric_constant import NO, YES
from openmind.rhetoric.model.tell import Tell


class StructuredTextEncoder:
    """The baseline data-to-text encoder: says a message's positions in structured language, and strips everything the
    message itself doesn't say, so the tone, the addressee and the positions' importance stay behind.
    Tell(joe, disgust, [do I want his fish: no, is his fish disgusting: yes]) becomes
    `[{question: do I want his fish, answer: no}, {question: is his fish disgusting, answer: yes}]`. An answer of 1 is
    said yes, -1 no, and any other as its number. A special character in a question is escaped with a backslash."""

    @property
    def name(self) -> str:
        return STRUCTURED_LANGUAGE

    def encode(self, message: Tell) -> str:
        said = (
            f"{{{QUESTION_FIELD}: {self._escaped(position.question)}, {ANSWER_FIELD}: {self._answer(position.answer)}}}"
            for position in message.positions
        )
        return f"[{', '.join(said)}]"

    def _answer(self, answer: float) -> str:
        if answer == YES:
            return "yes"
        if answer == NO:
            return "no"
        return f"{answer:g}"

    def _escaped(self, text: str) -> str:
        return "".join(f"\\{character}" if character in SPECIAL_CHARACTERS else character for character in text)
