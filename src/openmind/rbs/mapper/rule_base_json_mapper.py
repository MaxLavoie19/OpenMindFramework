import json

from openmind.expression.mapper.expression_json_mapper import ExpressionJsonMapper
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase


class RuleBaseJsonMapper:
    """Maps a rule base to JSON text and back."""

    def __init__(self, expression_json_mapper: ExpressionJsonMapper) -> None:
        self._expression_json_mapper = expression_json_mapper

    def to_json(self, rule_base: RuleBase) -> str:
        return json.dumps(
            {
                "domain": rule_base.domain,
                "rules": [
                    {
                        "action": rule.action,
                        "conditions": [self._expression_json_mapper.to_data(condition) for condition in rule.conditions],
                        "expected_value": rule.expected_value,
                        "visits": rule.visits,
                    }
                    for rule in rule_base.rules
                ],
            },
            indent=2,
        )

    def from_json(self, text: str) -> RuleBase:
        data = json.loads(text)
        return RuleBase(
            data["domain"],
            tuple(
                Rule(
                    item["action"],
                    tuple(self._expression_json_mapper.from_data(condition) for condition in item["conditions"]),
                    item["expected_value"],
                    item["visits"],
                )
                for item in data["rules"]
            ),
        )
