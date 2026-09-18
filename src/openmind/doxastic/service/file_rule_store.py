import json
import logging
from collections.abc import Iterator
from pathlib import Path

from openmind.doxastic.mapper.rule_record_json_mapper import RuleRecordJsonMapper
from openmind.doxastic.model.rule_record import RuleRecord

logger = logging.getLogger(__name__)

#: What marks a line as a rule forgotten rather than a rule.
FORGOTTEN = "forgotten"


class FileRuleStore:
    """The rules the agent knows, kept as JSON lines in one file, in the order they were declared.

    The file is only ever appended to: declaring a rule whose id is already there writes it anew and the last line
    wins, which is how a rule's weight in a context changes, and forgetting one writes a line saying so. A rule's place
    is its byte offset. Another store takes its place behind `RuleStore` without the knowledge base knowing."""

    def __init__(self, path: Path, rule_record_json_mapper: RuleRecordJsonMapper | None = None) -> None:
        self._path = path
        self._mapper = RuleRecordJsonMapper() if rule_record_json_mapper is None else rule_record_json_mapper

    def append(self, rule: RuleRecord) -> object:
        """Writes the rule at the end of the file and gives back its byte offset."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = self._mapper.to_line(rule).encode("utf-8")
        with self._path.open("ab") as file:
            offset = file.tell()
            file.write(line + b"\n")
        return offset

    def read(self, place: object) -> RuleRecord:
        """The rule written at that offset."""
        with self._path.open("rb") as file:
            file.seek(int(place))  # type: ignore[arg-type]
            line = file.readline()
        return self._mapper.from_line(line.decode("utf-8"))

    def load(self) -> Iterator[tuple[object, RuleRecord]]:
        """Every rule still declared, with its offset, in the order the ids were first written: a rule written again
        gives its newest line, and a forgotten one is left out."""
        if not self._path.exists():
            return
        places: dict[str, int] = {}
        rules: dict[str, RuleRecord] = {}
        with self._path.open("rb") as file:
            offset = 0
            for line in file:
                written_at, offset = offset, offset + len(line)
                text = line.decode("utf-8").strip()
                if not text:
                    continue
                forgotten = self._forgotten(text)
                if forgotten is not None:
                    places.pop(forgotten, None)
                    rules.pop(forgotten, None)
                    continue
                rule = self._mapper.from_line(text)
                places[rule.id] = written_at
                rules[rule.id] = rule
        for rule_id, rule in rules.items():
            yield places[rule_id], rule

    def forget(self, rule_id: str) -> None:
        """Writes a line saying the rule is forgotten; it is left out from then on, and the line it was on stays."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({FORGOTTEN: rule_id}, separators=(",", ":")) + "\n")
        logger.info("Forgot rule %s", rule_id)

    def _forgotten(self, text: str) -> str | None:
        """The id a line says is forgotten, or None where the line is a rule."""
        if f'"{FORGOTTEN}"' not in text:
            return None
        forgotten = json.loads(text).get(FORGOTTEN)
        return None if forgotten is None else str(forgotten)
