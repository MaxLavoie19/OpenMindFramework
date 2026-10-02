import math

import pytest

from openmind.inference.service.mechanics import Mechanics
from openmind.inference.service.position_view import PositionView
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.model.state import State


def a_view(**models):
    return PositionView(Mechanics(StateNamespaceMapper(), MemoryMeter()), None, State.of(**models))


def test_a_model_is_read_as_an_attribute_where_python_allows_the_name():
    assert a_view(turn="white").turn == "white"


def test_a_model_whose_name_is_not_python_is_read_by_subscript():
    """A game names what it holds so a person can read it — "black may castle king side" — and those names are
    offered as readings. An expression is Python source and cannot say that as an attribute."""
    view = a_view(turn="white", **{"black may castle king side": True})

    assert view["black may castle king side"] is True
    assert view["turn"] == "white"


def test_a_model_the_position_has_not_got_says_so():
    with pytest.raises(KeyError):
        a_view(turn="white")["no such thing"]


def test_a_look_ahead_over_one_square_reads_as_nothing_where_there_is_no_such_move(game) -> None:
    """**Not nought, which was measured to be badly wrong.** Asked how many pieces are left after an exchange,
    a square with no recapture answered nought and read as though both sides had been wiped off the board.

    Nought is a real reading — a count that came to nothing — so a move that does not exist has to answer
    something else. It answers nothing at all, which is what this project means by a term that did not fire."""
    played = game("tictactoe")
    view = Mechanics(StateNamespaceMapper(), MemoryMeter()).view(played, played.start())

    found = view.worst_changing("X", "cell", (1, 1), lambda after: sum(1 for one in after.cell if one is None))

    assert not math.isnan(found), "X can play into the corner, so there is something after that move"
    assert math.isnan(
        view.worst_changing("X", "cell", (9, 9), lambda after: sum(1 for one in after.cell if one is None))
    ), "and no move reaches a square off the board, so nothing follows one"
