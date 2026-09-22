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
