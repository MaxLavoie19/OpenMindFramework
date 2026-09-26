from dataclasses import dataclass

import pytest

from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.domain import Domain
from openmind.structure.model.kind import Kind
from openmind.structure.model.record import Record
from openmind.structure.model.schema import ActionKind, GridKind, Schema


@dataclass(frozen=True, slots=True)
class Cell(Record):
    row: int
    column: int

COLOUR = Kind("colour", values=("white", "black"))
TYPE = Kind("type", values=("pawn", "knight", "bishop", "rook", "queen", "king"))
PIECE = Kind("piece", parts=(("colour", COLOUR), ("type", TYPE)))
SQUARE = Kind("square", parts=(("colour", COLOUR), ("piece", PIECE.or_nothing())))
NAMES = CellNames(("a", "b"), ("2", "1"))
BOARD = GridKind(SQUARE, (2, 2), NAMES)


def a_schema():
    cells = Kind("cell", values=BOARD.cells())
    return Schema(
        models=(("grid", BOARD), ("turn", COLOUR)),
        actions=(ActionKind("move", (("origin", cells), ("destination", cells))),),
    )


def test_a_parameter_ranges_over_what_the_schema_says_and_not_over_what_has_been_seen():
    """A learner working from what a game happened to show it learns the positions it has seen. The whole space of
    candidates is laid out before a single move is looked at."""
    domains = a_schema().domains("move")

    assert domains["origin"] == ("a2", "b2", "a1", "b1")
    assert domains["destination"] == domains["origin"]


def test_a_closed_set_is_known_before_it_has_all_been_seen():
    """Six types are six types the first time one of them turns up, which is what lets "any type but a rook" be
    said rather than guessed at once the other five have appeared."""
    assert len(SQUARE.part("piece").part("type").domain) == 6


def test_a_part_that_may_be_absent_says_so_rather_than_inventing_a_value_for_it():
    """An empty square holds no piece. A game should not have to invent a piece meaning "none" to say that."""
    assert None in SQUARE.part("piece").domain
    assert None not in SQUARE.part("colour").domain




def test_a_grid_with_no_names_offers_no_cells_to_point_at():
    """A game that does not name its cells addresses them some other way, and a parameter cannot be one."""
    assert GridKind(SQUARE, (2, 2)).cells() == ()


def test_asking_for_something_the_schema_has_not_got_says_so():
    with pytest.raises(KeyError):
        a_schema().model("no such thing")
    with pytest.raises(KeyError):
        a_schema().action("no such thing")
    with pytest.raises(KeyError):
        SQUARE.part("no such thing")


def test_a_parameter_of_parts_ranges_over_every_combination_of_them():
    """This version of chess says a move is `move(origin(row, column), destination(row, column))`. Two ranges of
    eight give sixty-four cells to point at, and nothing had to know that a board is square."""
    from dataclasses import dataclass

    from openmind.structure.model.record import Record

    @dataclass(frozen=True, slots=True)
    class Cell(Record):
        row: int
        column: int

    rows = Kind("row", values=tuple(range(1, 9)))
    cell = Kind("cell", parts=(("row", rows), ("column", rows)), builds=Cell)

    assert len(cell.domain) == 64
    assert Cell(1, 1) in cell.domain and Cell(8, 8) in cell.domain


def test_a_kind_of_parts_the_game_never_said_how_to_make_hands_nothing_out():
    """It is a description of what such a thing has, not something OMF can build one of."""
    assert PIECE.domain == ()


ROW = Kind("row", values=(1, 2, 3, 4))
CELL = Kind("cell", parts=(("row", ROW), ("column", ROW)), builds=Cell)


def a_mover(shape=(4, 4), parameters=None, models=None):
    """A made-up game whose move is made by whatever stands somewhere, by so much in each direction."""
    return Schema(
        models=models if models is not None else (("grid", GridKind(SQUARE.or_nothing(), shape)),),
        actions=(
            ActionKind(
                "move",
                parameters or (("self", CELL), ("x", Domain.WHOLE), ("y", Domain.WHOLE)),
            ),
        ),
    )


def test_the_thing_that_acts_is_a_parameter_like_the_rest():
    """It points at a place, so which rules apply is settled inside the constraint problem rather than by
    something dispatching outside it — and what moves knows where it is without anything storing it twice."""
    held = a_mover().domains("move")

    assert len(held["self"]) == 16
    assert Cell(1, 1) in held["self"]


def test_numbers_the_game_left_unbounded_are_bounded_by_its_own_grid():
    """A game says whole numbers and says nothing about how far, which is what leaves the edge of the board to be
    learned rather than handed over. Listing them still needs a bound, and the grid is where it comes from."""
    held = a_mover().domains("move")

    assert held["x"] == tuple(range(-3, 4))
    assert held["y"] == held["x"]


def test_a_game_with_no_grid_keeps_its_numbers_unlistable():
    """Rather than a number somebody invented. Whoever tries to enumerate gets nothing, and can say so."""
    held = a_mover(models=(("turn", COLOUR),), parameters=(("x", Domain.WHOLE),))

    assert held.domains("move")["x"] == ()


def test_a_bound_the_game_did_give_is_not_widened_by_the_grid():
    """Narrowing only ever narrows. A game that said its numbers run one to two meant it."""
    held = a_mover(parameters=(("x", Domain.WHOLE.within(1, 2)),))

    assert held.domains("move")["x"] == (1, 2)


def test_the_widest_grid_says_how_far():
    """A game of several grids is bounded by the one reaching furthest, since anything nearer is covered by it."""
    held = a_mover(
        models=(
            ("grid", GridKind(SQUARE.or_nothing(), (3, 3))),
            ("bigger", GridKind(SQUARE.or_nothing(), (6, 6))),
        ),
        parameters=(("x", Domain.WHOLE),),
    )

    assert held.domains("move")["x"] == tuple(range(-5, 6))
