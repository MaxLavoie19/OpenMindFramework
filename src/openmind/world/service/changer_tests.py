import pytest

from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.world.model.change import Moved, Placed, Removed, Told
from openmind.world.model.state import State
from openmind.world.service.changer import Changer

NAMES = CellNames(("a", "b"), ("2", "1"))


def a_board():
    return State.of(
        piece=Grid.of([["king", "pawn"], [None, "rook"]], NAMES),
        color=Grid.of([["white", "black"], [None, "white"]], NAMES),
        turn="white",
    )


def test_something_carried_from_one_square_to_another_leaves_nothing_behind():
    left = Changer().applied(a_board(), [Moved("piece", "b1", "b2")])

    assert left.model("piece").at("b2") == "rook"
    assert left.model("piece").at("b1") is None


def test_what_is_removed_before_a_move_onto_its_square_is_taken():
    left = Changer().applied(a_board(), [Removed("piece", "b2"), Moved("piece", "b1", "b2")])

    assert left.model("piece").at("b2") == "rook"


def test_what_is_removed_after_a_move_onto_its_square_is_the_piece_that_arrived():
    """The order is the whole of it: the same two changes the other way round take the wrong piece."""
    left = Changer().applied(a_board(), [Moved("piece", "b1", "b2"), Removed("piece", "b2")])

    assert left.model("piece").at("b2") is None


def test_a_piece_taken_in_passing_stands_on_neither_square_the_move_names():
    left = Changer().applied(a_board(), [Removed("piece", "b2"), Moved("piece", "b1", "a2")])

    assert left.model("piece").at("a2") == "rook"
    assert left.model("piece").at("b2") is None
    assert left.model("piece").at("b1") is None


def test_something_put_down_need_not_be_what_was_carried():
    left = Changer().applied(a_board(), [Removed("piece", "b1"), Placed("piece", "b2", "queen")])

    assert left.model("piece").at("b2") == "queen"


def test_a_scalar_of_the_position_reads_otherwise():
    left = Changer().applied(a_board(), [Told("turn", "black")])

    assert left.value("turn") == "black"


def a_bigger_board(pieces, colors, turn="white"):
    wider = CellNames(("a", "b", "c"), ("3", "2", "1"))
    return State.of(piece=Grid.of(pieces, wider), color=Grid.of(colors, wider), turn=turn)


def test_a_piece_carried_across_is_read_as_one_thing_moved():
    before = a_bigger_board([[None, None, None], [None, None, None], ["rook", None, None]], [[None] * 3, [None] * 3, ["white", None, None]])
    after = a_bigger_board([[None, None, None], [None, None, None], [None, None, "rook"]], [[None] * 3, [None] * 3, [None, None, "white"]], "black")

    changes = Changer().between(before, after)

    assert Moved("piece", (3, 1), (3, 3)) in changes
    assert Told("turn", "black") in changes


def test_a_piece_taken_is_read_as_a_removal_before_the_move():
    before = a_bigger_board([[None, None, None], [None, "pawn", None], ["rook", None, None]], [[None] * 3, [None, "black", None], ["white", None, None]])
    after = a_bigger_board([[None, None, None], [None, "rook", None], [None, None, None]], [[None] * 3, [None, "white", None], [None] * 3], "black")

    changes = Changer().between(before, after)
    pieces = [one for one in changes if getattr(one, "model", None) == "piece"]

    assert pieces == [Removed("piece", (2, 2)), Moved("piece", (3, 1), (2, 2))]
    assert Changer().applied(before, changes) == after


def test_a_piece_taken_in_passing_is_removed_from_a_square_the_move_never_names():
    """The fact that cannot be had from the two squares of the action."""
    before = a_bigger_board([[None, None, None], ["pawn", "pawn", None], [None, None, None]], [[None] * 3, ["white", "black", None], [None] * 3])
    after = a_bigger_board([[None, "pawn", None], [None, None, None], [None, None, None]], [[None, "white", None], [None] * 3, [None] * 3], "black")

    changes = Changer().between(before, after)

    assert Removed("piece", (2, 2)) in changes
    assert Moved("piece", (2, 1), (1, 2)) in changes


def test_a_pawn_promoting_puts_down_something_other_than_what_it_carried():
    before = a_bigger_board([[None, None, None], ["pawn", None, None], [None, None, None]], [[None] * 3, ["white", None, None], [None] * 3])
    after = a_bigger_board([["queen", None, None], [None, None, None], [None, None, None]], [["white", None, None], [None] * 3, [None] * 3], "black")

    changes = Changer().between(before, after)
    pieces = [one for one in changes if getattr(one, "model", None) == "piece"]

    assert pieces == [Removed("piece", (2, 1)), Placed("piece", (1, 1), "queen")]


def test_a_rook_taking_a_rook_shows_in_the_colours_and_not_in_the_kinds():
    """A game that keeps what a thing is apart from whose it is hides a capture between alike pieces in one grid
    and shows it in the other: the kinds never changed on the square, so the only thing that grid can say is that
    a rook is gone from where it stood."""
    before = a_bigger_board([[None, "rook", None], [None, None, None], [None, "rook", None]], [[None, "black", None], [None] * 3, [None, "white", None]])
    after = a_bigger_board([[None, "rook", None], [None, None, None], [None, None, None]], [[None, "white", None], [None] * 3, [None] * 3], "black")

    changes = Changer().between(before, after)

    assert [one for one in changes if getattr(one, "model", None) == "piece"] == [Removed("piece", (3, 2))]
    assert Removed("color", (1, 2)) in changes
    assert Moved("color", (3, 2), (1, 2)) in changes
    assert Changer().applied(before, changes) == after


def test_what_is_read_between_two_positions_makes_the_second_one():
    before = a_bigger_board([[None, None, None], [None, "pawn", None], ["rook", None, None]], [[None] * 3, [None, "black", None], ["white", None, None]])
    after = a_bigger_board([[None, None, None], [None, "rook", None], [None, None, None]], [[None] * 3, [None, "white", None], [None] * 3], "black")

    assert Changer().applied(before, Changer().between(before, after)) == after


def test_changing_cells_of_something_that_holds_none_says_so():
    with pytest.raises(TypeError):
        Changer().applied(a_board(), [Removed("turn", "b2")])
