import json
from datetime import datetime

from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance
from openmind.doxastic.model.record import Record
from openmind.rbs.model.python_rule import PythonRule


class RecordJsonMapper:
    """Maps a record to one line of JSON and back, so a store can keep it word for word.

    A claim's rule is written as Python source where it is source; where the project gave a function instead, its name
    is written for the reader and the rule comes back None, since a function belongs to the project and not to the
    file. Everything else comes back as it went in."""

    def to_line(self, record: Record) -> str:
        """The record as one line of JSON, without a line break of its own."""
        return json.dumps(self.to_data(record), separators=(",", ":"))

    def from_line(self, line: str) -> Record:
        return self.from_data(json.loads(line))

    def to_data(self, record: Record) -> dict[str, object]:
        provenance = record.provenance
        data: dict[str, object] = {
            "id": record.id,
            "text": record.text,
            "source": provenance.source,
            "told": provenance.told,
            "at": provenance.at,
            "game": provenance.game,
            "round": provenance.round,
            "ply": provenance.ply,
            "when": None if provenance.when is None else provenance.when.isoformat(),
            "subjects": list(record.subjects),
            "names": list(record.names),
            "keywords": list(record.keywords),
            "supports": record.supports,
            "strength": record.strength,
            "count": record.count,
        }
        if record.claim is not None:
            data["claim"] = self._claim_data(record.claim)
        return data

    def from_data(self, data: dict[str, object]) -> Record:
        when = data.get("when")
        provenance = Provenance(
            str(data["source"]),
            data.get("told"),  # type: ignore[arg-type]
            data.get("at"),  # type: ignore[arg-type]
            data.get("game"),  # type: ignore[arg-type]
            data.get("round"),  # type: ignore[arg-type]
            data.get("ply"),  # type: ignore[arg-type]
            None if when is None else datetime.fromisoformat(str(when)),
        )
        claim = data.get("claim")
        return Record(
            str(data["text"]),
            provenance,
            tuple(data.get("subjects", ())),  # type: ignore[arg-type]
            tuple(data.get("names", ())),  # type: ignore[arg-type]
            tuple(data.get("keywords", ())),  # type: ignore[arg-type]
            None if claim is None else self._claim(claim),  # type: ignore[arg-type]
            data.get("supports"),  # type: ignore[arg-type]
            float(data.get("strength", 1.0)),  # type: ignore[arg-type]
            int(data.get("count", 1)),  # type: ignore[arg-type]
            str(data.get("id", "")),
        )

    def _claim_data(self, claim: Claim) -> dict[str, object]:
        rule = claim.rule
        return {
            "name": claim.name,
            "rule": rule.source if isinstance(rule, PythonRule) else None,
            "function": None if rule is None or isinstance(rule, PythonRule) else getattr(rule, "__qualname__", str(rule)),
            "about": list(claim.about),
            "holder": list(claim.holder),
        }

    def _claim(self, data: dict[str, object]) -> Claim:
        rule = data.get("rule")
        return Claim(
            str(data["name"]),
            None if rule is None else PythonRule(str(rule)),
            tuple(data.get("about", ())),  # type: ignore[arg-type]
            tuple(data.get("holder", ())),  # type: ignore[arg-type]
        )
