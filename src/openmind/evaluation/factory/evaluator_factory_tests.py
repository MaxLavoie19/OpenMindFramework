from openmind.evaluation.factory.evaluator_factory import create_evaluator
from openmind.evaluation.service.evaluator import Evaluator


def test_create_evaluator_gives_an_evaluator() -> None:
    assert isinstance(create_evaluator(), Evaluator)
