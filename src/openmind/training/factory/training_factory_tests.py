from openmind.training.factory.training_factory import (
    create_distiller,
    create_value_distiller,
    create_value_training_loop,
)
from openmind.training.service.distiller import Distiller
from openmind.training.service.value_distiller import ValueDistiller
from openmind.training.service.value_training_loop import ValueTrainingLoop


def test_create_distiller_gives_a_distiller() -> None:
    assert isinstance(create_distiller(), Distiller)


def test_create_value_distiller_gives_a_value_distiller() -> None:
    assert isinstance(create_value_distiller(), ValueDistiller)


def test_create_value_training_loop_gives_a_value_training_loop() -> None:
    assert isinstance(create_value_training_loop(), ValueTrainingLoop)
