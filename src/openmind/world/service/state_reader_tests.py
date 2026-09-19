import pytest

from openmind.structure.model.map import Map
from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

PLAYERS = Players(("X", "O"), "turn", "payoff")
READER = StateReader()


def test_value_reads_a_scalar() -> None:
    assert READER.value(State.of(turn="O"), "turn") == "O"


def test_an_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        READER.value(State.of(turn="O"), "missing")


def test_the_player_to_act_is_the_index_of_the_player_the_scalar_names() -> None:
    assert READER.player_to_act(State.of(turn="O"), PLAYERS) == 1


def test_a_name_that_is_not_a_player_is_refused() -> None:
    with pytest.raises(ValueError, match="not one of"):
        READER.player_to_act(State.of(turn="Z"), PLAYERS)


def test_players_flagged_in_a_map_act_at_once() -> None:
    state = State.of(turn=Map.of({"X": True, "O": True}))

    assert READER.acts_at_once(state, PLAYERS)
    assert READER.players_to_act(state, PLAYERS) == (0, 1)
    assert not READER.acts_at_once(State.of(turn="X"), PLAYERS)


def test_the_player_to_act_is_the_one_flagged_and_players_acting_at_once_raise() -> None:
    assert READER.player_to_act(State.of(turn=Map.of({"X": False, "O": True})), PLAYERS) == 1
    with pytest.raises(ValueError, match="act at once"):
        READER.player_to_act(State.of(turn=Map.of({"X": True, "O": True})), PLAYERS)


def test_payoffs_follow_the_order_of_players() -> None:
    assert READER.payoffs(State.of(payoff=Map.of({"O": 0.0, "X": 1.0})), PLAYERS) == (1.0, 0.0)


def test_a_payoff_that_is_not_a_number_raises() -> None:
    with pytest.raises(ValueError, match="not a number"):
        READER.payoffs(State.of(payoff=Map.of({"O": None, "X": 1.0})), PLAYERS)
