from openmind.rbs.mapper.value_rule_text_mapper import ValueRuleTextMapper
from openmind.rbs.model.value_rule import ValueRule
from openmind.rule.model.python_rule import PythonRule


def test_the_signed_weight_comes_before_the_term() -> None:
    mapper = ValueRuleTextMapper()

    assert mapper.to_text(ValueRule(PythonRule("wins(me)"), 0.42)) == "+0.42 × wins(me)"
    assert mapper.to_text(ValueRule(PythonRule("cell[2, 2] == other"), -1.2345678)) == "-1.23457 × cell[2, 2] == other"
