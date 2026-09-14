from openmind.training.builder.value_distiller_builder import ValueDistillerBuilder
from openmind.training.service.value_distiller import ValueDistiller


def test_build_gives_a_value_distiller() -> None:
    assert isinstance(ValueDistillerBuilder().with_workers(2).build(), ValueDistiller)
