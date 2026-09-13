from openmind.rbs.builder.rule_inducer_builder import RuleInducerBuilder
from openmind.rbs.service.rule_inducer import RuleInducer


def test_build_gives_a_rule_inducer() -> None:
    assert isinstance(RuleInducerBuilder().build(), RuleInducer)
