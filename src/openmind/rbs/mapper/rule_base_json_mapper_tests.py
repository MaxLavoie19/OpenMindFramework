import json

from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase
from openmind.rule.model.python_rule import PythonRule

RULE_BASE = RuleBase(
    "tictactoe",
    (
        Rule("place", (), 0.5, 1000),
        Rule("place", (PythonRule("cell[2, 2] == None"), PythonRule("row == 2")), 0.75, 400, priority=True),
    ),
)


def test_conditions_are_stored_as_their_source() -> None:
    assert json.loads(RuleBaseJsonMapper().to_json(RULE_BASE))["rules"][1] == {
        "action": "place",
        "conditions": ["cell[2, 2] == None", "row == 2"],
        "expected_value": 0.75,
        "visits": 400,
        "priority": True,
    }


def test_round_trip() -> None:
    mapper = RuleBaseJsonMapper()

    assert mapper.from_json(mapper.to_json(RULE_BASE)) == RULE_BASE


def test_a_rule_saved_without_priority_is_not_a_priority_rule() -> None:
    text = '{"domain": "d", "rules": [{"action": "a", "conditions": [], "expected_value": 0.5, "visits": 3}]}'

    assert RuleBaseJsonMapper().from_json(text).rules[0].priority is False
