import pytest

from openmind.structure.model.map import Map


def test_a_map_is_sorted_by_key_so_equal_contents_are_equal_maps() -> None:
    payoff = Map.of({"X": None, "O": None})

    assert payoff == Map.of({"O": None, "X": None}) and hash(payoff) == hash(Map.of({"O": None, "X": None}))
    assert payoff.keys() == ("O", "X")


def test_a_map_is_read_by_key_and_changes_into_a_new_map() -> None:
    payoff = Map.of({"X": None, "O": None})

    won = payoff.with_item("X", 1.0)

    assert (won["X"], payoff["X"], won.get("missing", 0)) == (1.0, None, 0)
    with pytest.raises(KeyError):
        payoff["missing"]
