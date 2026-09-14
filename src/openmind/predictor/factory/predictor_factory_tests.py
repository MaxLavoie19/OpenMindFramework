from openmind.predictor.factory.predictor_factory import create_predictor
from openmind.predictor.service.predictor import Predictor


def test_create_predictor_gives_a_predictor() -> None:
    assert isinstance(create_predictor(), Predictor)
