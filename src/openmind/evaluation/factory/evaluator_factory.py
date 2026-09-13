from openmind.evaluation.builder.evaluator_builder import EvaluatorBuilder
from openmind.evaluation.service.evaluator import Evaluator


def create_evaluator() -> Evaluator:
    """An evaluator with its match runner, exact search and solver."""
    return EvaluatorBuilder().build()
