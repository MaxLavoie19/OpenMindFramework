import json

from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rule.model.python_rule import PythonRule


class ValueBaseJsonMapper:
    """Maps a value base to JSON text and back; each term is stored as its Python source."""

    def to_json(self, value_base: ValueBase) -> str:
        return json.dumps(
            {
                "domain": value_base.domain,
                "bias": value_base.bias,
                "low": value_base.low,
                "high": value_base.high,
                "rules": [{"term": rule.term.source, "weight": rule.weight} for rule in value_base.rules],
            },
            indent=2,
        )

    def from_json(self, text: str) -> ValueBase:
        data = json.loads(text)
        return ValueBase(
            data["domain"],
            data["bias"],
            data["low"],
            data["high"],
            tuple(ValueRule(PythonRule(item["term"]), item["weight"]) for item in data["rules"]),
        )
