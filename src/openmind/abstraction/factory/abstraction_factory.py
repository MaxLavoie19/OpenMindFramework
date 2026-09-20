from openmind.abstraction.service.rule_abstractor import RuleAbstractor
from openmind.rule.factory.rule_factory import create_rule_caller


def create_rule_abstractor() -> RuleAbstractor:
    """The service running an abstraction ruleset's RBS, with its own rule caller. Build it once and give it to
    whatever holds a level of the hierarchy."""
    return RuleAbstractor(create_rule_caller())
