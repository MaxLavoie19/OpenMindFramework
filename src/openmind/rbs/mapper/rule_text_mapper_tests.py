from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.model.rule import Rule
from openmind.rule.model.python_rule import PythonRule


def test_a_rule_with_conditions() -> None:
    rule = Rule("place", (PythonRule("cell[row, col] == None"), PythonRule("cell[2, 2] == None")), 0.75, 400)

    assert RuleTextMapper().to_text(rule) == (
        "place when cell[row, col] == None and cell[2, 2] == None: EV 0.75 over 400 visits"
    )


def test_a_rule_without_conditions() -> None:
    assert RuleTextMapper().to_text(Rule("place", (), 0.5, 1000)) == "place in any state: EV 0.5 over 1000 visits"


def test_a_priority_rule_says_so() -> None:
    rule = Rule("place", (PythonRule("win_chance(action) >= 1"),), 1.0, 90, priority=True)

    assert RuleTextMapper().to_text(rule) == "place when win_chance(action) >= 1: EV 1.0 over 90 visits, priority"
