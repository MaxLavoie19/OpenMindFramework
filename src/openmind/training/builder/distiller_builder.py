from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.rbs.builder.rule_inducer_builder import RuleInducerBuilder
from openmind.training.service.distiller import Distiller
from openmind.training.service.self_play import SelfPlay
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class DistillerBuilder:
    """Wires the services a distiller works with."""

    def build(self) -> Distiller:
        names = VariableNameMapper()
        interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
        self_play = SelfPlay(
            SolverBuilder().build(),
            Predictor(interpreter, names, expression_text, action_text),
            StateReader(),
        )
        return Distiller(self_play, RuleInducerBuilder().build(), interpreter)
