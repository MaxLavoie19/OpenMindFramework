from openmind.agent.model.domain import Domain
from openmind.rbs.builder.rule_generator_builder import RuleGeneratorBuilder
from openmind.rbs.builder.rule_rater_builder import RuleRaterBuilder
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.rule_generator import RuleGenerator
from openmind.rbs.service.rule_rater import RuleRater


def create_rule_generator() -> RuleGenerator:
    """A rule generator with its goal pattern miner, primitive generator, hypothesis discoverer and validator."""
    return RuleGeneratorBuilder().build()


def create_rule_rater(rule_base: RuleBase, domain: Domain) -> RuleRater:
    """A rater rating the domain's actions with the given rule base."""
    return RuleRaterBuilder().with_rule_base(rule_base).with_domain(domain).build()
