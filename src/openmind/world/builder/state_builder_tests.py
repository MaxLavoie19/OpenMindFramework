import pytest

from openmind.structure.model.grid import Grid
from openmind.world.builder.state_builder import StateBuilder
from openmind.world.model.state import State


def test_the_builder_collects_models_into_a_state() -> None:
    board = Grid.filled((3, 3), None)

    assert StateBuilder().with_model("turn", "X").with_model("cell", board).build() == State.of(cell=board, turn="X")


def test_a_model_set_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="already set"):
        StateBuilder().with_model("turn", "X").with_model("turn", "O")
