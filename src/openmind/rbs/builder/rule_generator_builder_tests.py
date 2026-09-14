from openmind.rbs.builder.rule_generator_builder import RuleGeneratorBuilder
from openmind.rbs.service.rule_generator import RuleGenerator


def test_build_gives_a_rule_generator() -> None:
    assert isinstance(RuleGeneratorBuilder().build(), RuleGenerator)
