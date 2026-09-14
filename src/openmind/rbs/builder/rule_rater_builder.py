from typing import Self

from openmind.agent.model.domain import Domain
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.rule_base import RuleBase
from openmind.rbs.service.rule_rater import RuleRater
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class RuleRaterBuilder:
    """Sets the rule base a rater rates with and the domain it rates in, and wires its rule compiler, rule runner and
    consequence library; rejects a missing rule base or domain."""

    def __init__(self) -> None:
        self._rule_base: RuleBase | None = None
        self._domain: Domain | None = None

    def with_rule_base(self, rule_base: RuleBase) -> Self:
        self._rule_base = rule_base
        return self

    def with_domain(self, domain: Domain) -> Self:
        self._domain = domain
        return self

    def build(self) -> RuleRater:
        if self._rule_base is None:
            raise ValueError("Rule rater needs a rule base")
        if self._domain is None:
            raise ValueError("Rule rater needs a domain")
        return RuleRater(
            self._rule_base,
            self._domain,
            RuleCompiler(),
            RuleRunner(StateNamespaceMapper(VariableNameMapper())),
            ConsequenceLibraryBuilder().build(),
        )
