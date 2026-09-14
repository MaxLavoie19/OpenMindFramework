from typing import Self

from openmind.agent.model.domain import Domain
from openmind.rbs.builder.consequence_library_builder import ConsequenceLibraryBuilder
from openmind.rbs.model.value_base import ValueBase
from openmind.rbs.service.rule_valuer import RuleValuer
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class RuleValuerBuilder:
    """Sets the value base a valuer values with and the domain it values positions of, and wires its rule compiler, rule
    runner and consequence library; rejects a missing value base or domain."""

    def __init__(self) -> None:
        self._value_base: ValueBase | None = None
        self._domain: Domain | None = None

    def with_value_base(self, value_base: ValueBase) -> Self:
        self._value_base = value_base
        return self

    def with_domain(self, domain: Domain) -> Self:
        self._domain = domain
        return self

    def build(self) -> RuleValuer:
        if self._value_base is None:
            raise ValueError("Rule valuer needs a value base")
        if self._domain is None:
            raise ValueError("Rule valuer needs a domain")
        return RuleValuer(
            self._value_base,
            self._domain,
            RuleCompiler(),
            RuleRunner(StateNamespaceMapper(VariableNameMapper())),
            ConsequenceLibraryBuilder().build(),
        )
