from openmind.training.builder.distiller_builder import DistillerBuilder
from openmind.training.service.distiller import Distiller


def create_distiller(workers: int = 1) -> Distiller:
    """A distiller with its self-play, running games in that many worker processes, its rule generator, and the rater
    services it measures rules with."""
    return DistillerBuilder().with_workers(workers).build()
