from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.predictor.service.predictor import Predictor


def test_build_gives_a_predictor() -> None:
    assert isinstance(PredictorBuilder().build(), Predictor)
