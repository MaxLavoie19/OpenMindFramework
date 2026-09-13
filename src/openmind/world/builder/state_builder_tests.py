import pytest

from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.state import State


def test_build_sorts_variables_by_name() -> None:
    state = StateBuilder().with_variable("turn", "X").with_variable("cell(1,1)", None).build()

    assert state == State((("cell(1,1)", None), ("turn", "X")))


def test_build_without_variables_gives_an_empty_state() -> None:
    assert StateBuilder().build() == State(())


def test_with_variable_rejects_a_name_already_set() -> None:
    builder = StateBuilder().with_variable("turn", "X")

    with pytest.raises(ValueError, match="turn"):
        builder.with_variable("turn", "O")
