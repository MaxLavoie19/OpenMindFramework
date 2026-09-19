import pytest

from openmind.structure.model.map import Map
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

PLAYERS = Players(("X", "O"), "payoff")
READER = StateReader()


def test_value_reads_a_scalar() -> None:
    assert READER.value(State.of(turn="O"), "turn") == "O"


def test_an_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        READER.value(State.of(turn="O"), "missing")


def test_payoffs_follow_the_order_of_players() -> None:
    assert READER.payoffs(State.of(payoff=Map.of({"O": 0.0, "X": 1.0})), PLAYERS) == (1.0, 0.0)


def test_a_payoff_that_is_not_a_number_raises() -> None:
    with pytest.raises(ValueError, match="not a number"):
        READER.payoffs(State.of(payoff=Map.of({"O": None, "X": 1.0})), PLAYERS)
