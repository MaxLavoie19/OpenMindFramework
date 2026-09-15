from openmind.observation.service.state_observer import StateObserver
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class StateObserverBuilder:
    """Wires a state observer with its rule compiler and rule runner."""

    def build(self) -> StateObserver:
        return StateObserver(RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())))
