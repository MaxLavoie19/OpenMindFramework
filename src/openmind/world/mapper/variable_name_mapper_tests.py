import pytest

from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.value import Value


@pytest.mark.parametrize(
    ("base", "indices", "name"),
    [
        ("turn", (), "turn"),
        ("cell", (2, 3), "cell(2,3)"),
        ("payoff", ("X",), "payoff(X)"),
    ],
)
def test_to_name(base: str, indices: tuple[Value, ...], name: str) -> None:
    assert VariableNameMapper().to_name(base, indices) == name


@pytest.mark.parametrize(
    ("name", "base", "indices"),
    [
        ("turn", "turn", ()),
        ("cell(2,3)", "cell", ("2", "3")),
        ("payoff(X)", "payoff", ("X",)),
    ],
)
def test_from_name(name: str, base: str, indices: tuple[str, ...]) -> None:
    assert VariableNameMapper().from_name(name) == (base, indices)
