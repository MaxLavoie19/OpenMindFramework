from openmind.language.constant.structured_language_constant import ANSWER_FIELD, QUESTION_FIELD, STRUCTURED_LANGUAGE
from openmind.rhetoric.constant.rhetoric_constant import NO, YES
from openmind.rhetoric.model.position import Position


class StructuredTextDecoder:
    """The baseline text-to-data decoder: reads the positions structured language says back into data, as the agent
    hearing the message gets them, without importance. Text that isn't structured language raises ValueError."""

    @property
    def name(self) -> str:
        return STRUCTURED_LANGUAGE

    def decode(self, text: str) -> tuple[Position, ...]:
        body = text.strip()
        if not (body.startswith("[") and body.endswith("]")):
            raise ValueError(f"Structured language is a list in brackets, not {text!r}")
        positions: list[Position] = []
        for item in self._split(body[1:-1], ","):
            item = item.strip()
            if not item:
                continue
            if not (item.startswith("{") and item.endswith("}")):
                raise ValueError(f"A position is written in braces, not {item!r}")
            fields: dict[str, str] = {}
            for field in self._split(item[1:-1], ","):
                name, separator, value = self._partition(field)
                if not separator:
                    raise ValueError(f"A field is a name, a colon and a value, not {field!r}")
                fields[name.strip()] = self._unescaped(value.strip())
            if set(fields) != {QUESTION_FIELD, ANSWER_FIELD}:
                raise ValueError(f"A position has a question and an answer, not {sorted(fields)}")
            positions.append(Position(fields[QUESTION_FIELD], self._answer(fields[ANSWER_FIELD])))
        return tuple(positions)

    def _answer(self, text: str) -> float:
        """yes is 1, no is -1, and a number is itself; anything else raises ValueError."""
        words = {"yes": YES, "no": NO}
        return words[text] if text in words else float(text)

    def _split(self, text: str, separator: str) -> list[str]:
        """Splits at the separator where it isn't escaped or inside braces."""
        parts, current, depth, escaped = [], [], 0, False
        for character in text:
            if escaped:
                current.append(f"\\{character}")
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == "{":
                depth += 1
                current.append(character)
            elif character == "}":
                depth -= 1
                current.append(character)
            elif character == separator and depth == 0:
                parts.append("".join(current))
                current = []
            else:
                current.append(character)
        parts.append("".join(current))
        return parts

    def _partition(self, field: str) -> tuple[str, str, str]:
        """The field's name, the first unescaped colon, and its value."""
        escaped = False
        for at, character in enumerate(field):
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == ":":
                return field[:at], ":", field[at + 1 :]
        return field, "", ""

    def _unescaped(self, text: str) -> str:
        characters, escaped = [], False
        for character in text:
            if escaped:
                characters.append(character)
                escaped = False
            elif character == "\\":
                escaped = True
            else:
                characters.append(character)
        return "".join(characters)
