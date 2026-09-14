from openmind.evaluation.builder.evaluator_builder import EvaluatorBuilder
from openmind.evaluation.service.evaluator import Evaluator


def create_evaluator(workers: int = 1) -> Evaluator:
    """An evaluator with its match runner, exact and reference searches, running games and searches in that many
    worker processes."""
    return EvaluatorBuilder().with_workers(workers).build()
