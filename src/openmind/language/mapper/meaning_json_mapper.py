import json

from openmind.language.model.meaning import Meaning


class MeaningJsonMapper:
    """Maps a meaning to one line of JSON, keys in the order the meaning holds them, and back: the form an encoder
    reads and training examples store."""

    def to_json(self, meaning: Meaning) -> str:
        return json.dumps(meaning, ensure_ascii=False, separators=(", ", ": "))

    def from_json(self, text: str) -> Meaning:
        meaning = json.loads(text)
        if not isinstance(meaning, dict):
            raise ValueError(f"A meaning is a JSON object, not {text!r}")
        return meaning
