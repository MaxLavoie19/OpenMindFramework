import json

from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rule.model.python_rule import PythonRule
from openmind.training.model.rule_support import RuleSupport
from openmind.training.model.signal import Signal
from openmind.training.model.signal_library import SignalLibrary
from openmind.training.model.signal_record import SignalRecord


class SignalLibraryJsonMapper:
    """Maps a signal library to JSON text and back: every record with its signal, sources as Python source, every rule
    support with its strength by signal, and every followed signal's value base as `ValueBaseJsonMapper` writes it."""

    def __init__(self, value_base_json_mapper: ValueBaseJsonMapper | None = None) -> None:
        self._value_base_json_mapper = ValueBaseJsonMapper() if value_base_json_mapper is None else value_base_json_mapper

    def to_json(self, library: SignalLibrary) -> str:
        return json.dumps(
            {
                "domain": library.domain,
                "records": [
                    {
                        "name": record.signal.name,
                        "source": None if record.signal.source is None else record.signal.source.source,
                        "parts": list(record.signal.parts),
                        "premises": list(record.signal.premises),
                        "agreements": record.agreements,
                        "disagreements": record.disagreements,
                        "accuracy": record.accuracy,
                        "reliability": record.reliability,
                        "games": record.games,
                        "wins": record.wins,
                        "draws": record.draws,
                        "losses": record.losses,
                    }
                    for record in library.records
                ],
                "supports": [
                    {
                        "term": support.term.source,
                        "strengths": [{"signal": name, "strength": strength} for name, strength in support.strengths],
                    }
                    for support in library.supports
                ],
                "value_bases": [
                    {"signal": name, "value_base": json.loads(self._value_base_json_mapper.to_json(value_base))}
                    for name, value_base in library.value_bases
                ],
            },
            indent=2,
        )

    def from_json(self, text: str) -> SignalLibrary:
        """Accuracy and reliability are read back from the counts, not from the text."""
        data = json.loads(text)
        records = tuple(
            SignalRecord(
                Signal(
                    item["name"],
                    None if item["source"] is None else PythonRule(item["source"]),
                    tuple(item["parts"]),
                    tuple(item.get("premises", [])),
                ),
                item["agreements"],
                item["disagreements"],
                item["games"],
                item["wins"],
                item["draws"],
                item["losses"],
            )
            for item in data["records"]
        )
        supports = tuple(
            RuleSupport(
                PythonRule(item["term"]),
                tuple((strength["signal"], strength["strength"]) for strength in item["strengths"]),
            )
            for item in data["supports"]
        )
        value_bases = tuple(
            (item["signal"], self._value_base_json_mapper.from_json(json.dumps(item["value_base"])))
            for item in data.get("value_bases", [])
        )
        return SignalLibrary(data["domain"], records, supports, value_bases)
