from openmind.agent.model.domain import Domain
from openmind.rbs.builder.rule_generator_builder import RuleGeneratorBuilder
from openmind.rbs.builder.rule_rater_builder import RuleRaterBuilder
from openmind.rbs.builder.rule_valuer_builder import RuleValuerBuilder
from openmind.rbs.builder.value_generator_builder import ValueGeneratorBuilder
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.service.rule_generator import RuleGenerator
from openmind.rbs.service.rule_rater import RuleRater
from openmind.rbs.service.rule_valuer import RuleValuer
from openmind.rbs.service.value_generator import ValueGenerator


def create_rule_generator(workers: int = 1) -> RuleGenerator:
    """A rule generator with its goal pattern miner, primitive generator, hypothesis discoverer and validator, checking
    conditions in that many worker processes."""
    return RuleGeneratorBuilder().with_workers(workers).build()


def create_rule_rater(rule_base: RuleBase, domain: Domain) -> RuleRater:
    """A rater rating the domain's actions with the given rule base."""
    return RuleRaterBuilder().with_rule_base(rule_base).with_domain(domain).build()


def create_value_generator(workers: int = 1) -> ValueGenerator:
    """A value generator with its term generator, term evaluator and sparse fitter, evaluating terms in that many worker
    processes."""
    return ValueGeneratorBuilder().with_workers(workers).build()


def create_rule_valuer(value_base: ValueBase, domain: Domain) -> RuleValuer:
    """A valuer valuing the domain's positions with the given value base."""
    return RuleValuerBuilder().with_value_base(value_base).with_domain(domain).build()
