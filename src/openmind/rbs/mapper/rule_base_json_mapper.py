import json

from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rule.model.python_rule import PythonRule


class RuleBaseJsonMapper:
    """Maps a rule base to JSON text and back; each condition is stored as its Python source. A rule saved without
    `priority` loads as not a priority rule."""

    def to_json(self, rule_base: RuleBase) -> str:
        return json.dumps(
            {
                "domain": rule_base.domain,
                "rules": [
                    {
                        "action": rule.action,
                        "conditions": [condition.source for condition in rule.conditions],
                        "expected_value": rule.expected_value,
                        "visits": rule.visits,
                        "priority": rule.priority,
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
                    tuple(PythonRule(source) for source in item["conditions"]),
                    item["expected_value"],
                    item["visits"],
                    item.get("priority", False),
                )
                for item in data["rules"]
            ),
        )
