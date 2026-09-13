from openmind.training.builder.distiller_builder import DistillerBuilder
from openmind.training.service.distiller import Distiller


def test_build_gives_a_distiller() -> None:
    assert isinstance(DistillerBuilder().build(), Distiller)
