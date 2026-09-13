from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.rbs.mapper.rule_text_mapper import RuleTextMapper
from openmind.rbs.service.rule_inducer import RuleInducer
from openmind.world.mapper.variable_name_mapper import VariableNameMapper


class RuleInducerBuilder:
    """Wires the services a rule inducer works with."""

    def build(self) -> RuleInducer:
        names = VariableNameMapper()
        return RuleInducer(Interpreter(names), names, RuleTextMapper(ExpressionTextMapper(names)))
