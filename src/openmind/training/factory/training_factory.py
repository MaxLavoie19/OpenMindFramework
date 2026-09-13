from openmind.training.builder.distiller_builder import DistillerBuilder
from openmind.training.service.distiller import Distiller


def create_distiller() -> Distiller:
    """A distiller with its self-play, rule inducer and interpreter."""
    return DistillerBuilder().build()
