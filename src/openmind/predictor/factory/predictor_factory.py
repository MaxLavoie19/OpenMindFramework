from openmind.predictor.builder.predictor_builder import PredictorBuilder
from openmind.predictor.service.predictor import Predictor


def create_predictor() -> Predictor:
    """A predictor running effects scripts with its own rule compiler and runner."""
    return PredictorBuilder().build()
