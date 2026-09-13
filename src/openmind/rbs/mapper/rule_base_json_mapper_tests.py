from openmind.expression.mapper.expression_json_mapper import ExpressionJsonMapper
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.rbs.mapper.rule_base_json_mapper import RuleBaseJsonMapper
from openmind.rbs.model.rule import Rule
from openmind.rbs.model.rule_base import RuleBase


def test_round_trip() -> None:
    rule_base = RuleBase(
        "tictactoe",
        (
            Rule("place", (), 0.5, 1000),
            Rule(
                "place",
                (
                    Equals(StateVariable("cell(2,2)"), Constant(None)),
                    Equals(ActionParameter("row"), Constant(2)),
                ),
                0.75,
                400,
            ),
        ),
    )
    mapper = RuleBaseJsonMapper(ExpressionJsonMapper())

    assert mapper.from_json(mapper.to_json(rule_base)) == rule_base
