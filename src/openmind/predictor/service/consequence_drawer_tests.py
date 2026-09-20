from openmind.predictor.model.consequence import Consequence
from openmind.predictor.model.drawn import Always, Asked, Column, More, Other, Row, Standing
from openmind.predictor.service.consequence_drawer import ConsequenceDrawer
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.world.model.action import Action
from openmind.world.model.change import Moved, Placed, Removed, Told
from openmind.world.model.state import State

NAMES = CellNames(("a", "b", "c"), ("3", "2", "1"))
PLAYERS = ("white", "black")


def a_board():
    return State.of(
        piece=Grid.of([[None, None, None], ["pawn", "pawn", None], [None, None, None]], NAMES),
        color=Grid.of([[None, None, None], ["white", "black", None], [None, None, None]], NAMES),
        turn="white",
        halfmove=7,
    )


def walking(promotion=None):
    return Action("move", (("source", "a2"), ("target", "b3"), ("promotion", promotion)))


def drawn(consequence, action=None):
    return ConsequenceDrawer().drawn(consequence, a_board(), action or walking(), "white", PLAYERS)


def test_a_square_is_drawn_from_the_rows_and_columns_of_the_action():
    """En passant: the piece taken stands at the row it started on and the column it lands on, and no parameter
    of the action names that square."""
    taken = Consequence("Removed", "piece", (Row("source"), Column("target")))

    assert drawn(taken) == Removed("piece", NAMES.to_coordinates("b2"))


def test_what_is_carried_goes_from_one_drawn_square_to_another():
    carried = Consequence("Moved", "piece", (Row("source"), Column("source")), (Row("target"), Column("target")))

    assert drawn(carried) == Moved("piece", NAMES.to_coordinates("a2"), NAMES.to_coordinates("b3"))


def test_what_is_put_down_can_be_what_the_action_asked_for():
    promoting = Consequence("Placed", "piece", (Row("target"), Column("target")), value=Asked("promotion"))

    assert drawn(promoting, walking("queen")) == Placed("piece", NAMES.to_coordinates("b3"), "queen")


def test_what_is_put_down_can_be_what_was_standing_where_it_started():
    carried = Consequence("Placed", "color", (Row("target"), Column("target")), value=Standing("color", "source"))

    assert drawn(carried) == Placed("color", NAMES.to_coordinates("b3"), "white")


def test_a_scalar_can_be_told_the_player_not_acting():
    assert drawn(Consequence("Told", "turn", value=Other())) == Told("turn", "black")


def test_a_scalar_can_be_told_what_it_read_and_one_more():
    assert drawn(Consequence("Told", "halfmove", value=More("halfmove"))) == Told("halfmove", 8)


def test_a_scalar_can_be_told_the_same_thing_whatever_the_action():
    assert drawn(Consequence("Told", "en_passant", value=Always(None))) == Told("en_passant", None)


def test_a_square_off_the_board_is_drawn_as_nothing():
    """A consequence that would name a square the board does not have says nothing rather than raising: a rule
    tried where it does not fit has to answer."""
    outside = Consequence("Removed", "piece", (Always(9), Always(9)))

    assert drawn(outside) is None
