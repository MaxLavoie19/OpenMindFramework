from openmind.predictor.service.predictor import Predictor
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.service.rule_compiler import RuleCompiler
from openmind.rule.service.rule_runner import RuleRunner
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class PredictorBuilder:
    """Wires a predictor with its rule compiler, rule runner and action text mapper."""

    def build(self) -> Predictor:
        return Predictor(RuleCompiler(), RuleRunner(StateNamespaceMapper(VariableNameMapper())), ActionTextMapper())
