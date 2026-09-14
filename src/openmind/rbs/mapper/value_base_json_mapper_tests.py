import json

from openmind.rbs.mapper.value_base_json_mapper import ValueBaseJsonMapper
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.model.value_rule import ValueRule
from openmind.rule.model.python_rule import PythonRule

VALUE_BASE = ValueBase(
    "tictactoe",
    -0.25,
    0.0,
    1.0,
    (ValueRule(PythonRule("wins(me)"), 0.42), ValueRule(PythonRule("cell[2, 2] == other"), -1.5)),
)


def test_terms_are_stored_as_their_source() -> None:
    assert json.loads(ValueBaseJsonMapper().to_json(VALUE_BASE)) == {
        "domain": "tictactoe",
        "bias": -0.25,
        "low": 0.0,
        "high": 1.0,
        "rules": [{"term": "wins(me)", "weight": 0.42}, {"term": "cell[2, 2] == other", "weight": -1.5}],
    }


def test_round_trip() -> None:
    mapper = ValueBaseJsonMapper()

    assert mapper.from_json(mapper.to_json(VALUE_BASE)) == VALUE_BASE
