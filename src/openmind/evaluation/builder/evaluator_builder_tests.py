from openmind.evaluation.builder.evaluator_builder import EvaluatorBuilder
from openmind.evaluation.service.evaluator import Evaluator


def test_build_gives_an_evaluator() -> None:
    assert isinstance(EvaluatorBuilder().build(), Evaluator)
