import pytest

from openmind.world.model.players import Players
from openmind.world.model.state import State
from openmind.world.service.state_reader import StateReader

PLAYERS = Players(("X", "O"), "turn", ("payoff(X)", "payoff(O)"))


def test_value_reads_a_variable() -> None:
    assert StateReader().value(State((("cell(1,1)", "X"), ("turn", "O"))), "turn") == "O"


def test_unknown_variable_raises() -> None:
    with pytest.raises(KeyError, match="speed"):
        StateReader().value(State(()), "speed")


def test_player_to_act_is_the_index_of_the_named_player() -> None:
    assert StateReader().player_to_act(State((("turn", "O"),)), PLAYERS) == 1


def test_player_to_act_rejects_a_name_that_is_not_a_player() -> None:
    with pytest.raises(ValueError, match="turn"):
        StateReader().player_to_act(State((("turn", "Z"),)), PLAYERS)


def test_payoffs_follow_the_order_of_players() -> None:
    state = State((("payoff(O)", 0), ("payoff(X)", 1.0)))

    assert StateReader().payoffs(state, PLAYERS) == (1.0, 0.0)


def test_a_payoff_that_is_not_a_number_raises() -> None:
    with pytest.raises(ValueError, match=r"payoff\(O\)"):
        StateReader().payoffs(State((("payoff(O)", None), ("payoff(X)", 1.0))), PLAYERS)
