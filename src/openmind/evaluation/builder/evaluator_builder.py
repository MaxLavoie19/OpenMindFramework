from openmind.csp.builder.solver_builder import SolverBuilder
from openmind.evaluation.service.evaluator import Evaluator
from openmind.evaluation.service.exact_search import ExactSearch
from openmind.evaluation.service.match_runner import MatchRunner
from openmind.expression.mapper.expression_text_mapper import ExpressionTextMapper
from openmind.expression.service.interpreter import Interpreter
from openmind.predictor.service.predictor import Predictor
from openmind.world.mapper.action_text_mapper import ActionTextMapper
from openmind.world.mapper.state_text_mapper import StateTextMapper
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.service.state_reader import StateReader


class EvaluatorBuilder:
    """Wires the services an evaluator measures with."""

    def build(self) -> Evaluator:
        names = VariableNameMapper()
        interpreter, expression_text, action_text = Interpreter(names), ExpressionTextMapper(names), ActionTextMapper()
        solver = SolverBuilder().build()
        predictor = Predictor(interpreter, names, expression_text, action_text)
        state_reader = StateReader()
        return Evaluator(
            MatchRunner(solver, predictor, state_reader),
            ExactSearch(solver, predictor, state_reader),
            solver,
            StateTextMapper(),
            action_text,
        )
