from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.model.action_parameter import ActionParameter
from openmind.expression.model.constant import Constant
from openmind.expression.model.equals import Equals
from openmind.expression.model.state_variable import StateVariable
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.rule import Rule
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


def test_a_rule_with_conditions() -> None:
    rule = Rule(
        "place",
        (
            Equals(StateVariable("cell", (ActionParameter("row"), ActionParameter("col"))), Constant(None)),
            Equals(StateVariable("cell(2,2)"), Constant(None)),
        ),
        0.75,
        400,
    )

    assert RuleTextMapper(ExpressionTextMapper(VariableNameMapper())).to_text(rule) == (
        "place when cell(row,col) == None and cell(2,2) == None: EV 0.75 over 400 visits"
    )


def test_a_rule_without_conditions() -> None:
    rule = Rule("place", (), 0.5, 1000)

    assert RuleTextMapper(ExpressionTextMapper(VariableNameMapper())).to_text(rule) == (
        "place in any state: EV 0.5 over 1000 visits"
    )
