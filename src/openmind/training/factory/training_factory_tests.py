from openmind.training.factory.training_factory import create_distiller
from openmind.training.service.distiller import Distiller


def test_create_distiller_gives_a_distiller() -> None:
    assert isinstance(create_distiller(), Distiller)
