from openmind.rbs.builder.value_generator_builder import ValueGeneratorBuilder
from openmind.rbs.service.value_generator import ValueGenerator


def test_build_gives_a_value_generator() -> None:
    assert isinstance(ValueGeneratorBuilder().with_workers(2).build(), ValueGenerator)
