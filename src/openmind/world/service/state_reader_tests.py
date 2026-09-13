import pytest

from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader


def test_value_reads_a_variable() -> None:
    assert StateReader().value(State((("cell(1,1)", "X"), ("turn", "O"))), "turn") == "O"


def test_unknown_variable_raises() -> None:
    with pytest.raises(KeyError, match="speed"):
        StateReader().value(State(()), "speed")
