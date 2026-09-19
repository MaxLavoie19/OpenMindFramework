from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.service.rule_caller import RuleCaller
from openmind.rbs.service.rule_compiler import RuleCompiler
from openmind.rbs.service.rule_runner import RuleRunner


def create_rule_caller() -> RuleCaller:
    """A rule caller with its own compiler and runner: it calls a domain's rules, written as Python source or as the
    project's own functions."""
    return RuleCaller(RuleCompiler(), RuleRunner(StateNamespaceMapper()))
