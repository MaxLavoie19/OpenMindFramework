from openmind.training.builder.rule_selector_builder import RuleSelectorBuilder
from openmind.training.service.rule_selector import RuleSelector


def test_build_gives_a_rule_selector() -> None:
    assert isinstance(RuleSelectorBuilder().with_workers(2).build(), RuleSelector)
