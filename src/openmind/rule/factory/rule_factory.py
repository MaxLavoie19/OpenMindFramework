from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_caller import RuleCaller
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


def create_rule_caller() -> RuleCaller:
    """A rule caller with its own compiler and runner: it calls a domain's rules, written as Python source or as the
    project's own functions."""
    return RuleCaller(RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())))
