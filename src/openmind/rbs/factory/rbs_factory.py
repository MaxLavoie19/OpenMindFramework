from openmind.rbs.builder.rule_inducer_builder import RuleInducerBuilder
from openmind.rbs.builder.rule_rater_builder import RuleRaterBuilder
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.rule_inducer import RuleInducer
from openmind.rbs.service.rule_rater import RuleRater


def create_rule_inducer() -> RuleInducer:
    """A rule inducer with its interpreter and mappers."""
    return RuleInducerBuilder().build()


def create_rule_rater(rule_base: RuleBase) -> RuleRater:
    """A rater rating actions with the given rule base."""
    return RuleRaterBuilder().with_rule_base(rule_base).build()
