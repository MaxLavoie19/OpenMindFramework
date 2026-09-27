from dataclasses import dataclass

from openmind.predictor.model.watched import Watched
from openmind.statement.model.consequence import Consequence
from openmind.statement.model.clause import Clause
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant
from openmind.predictor.service.consequence_learner import ConsequenceLearner


def test_a_condition_reads_as_a_condition_and_not_as_a_refusal():
    """Conditions are learned by the machinery that learns what a game refuses, so each carries that head — and
    "this happens where refused :- the turn is white" is not what it says. What holds is the conditions; what
    they conclude here is that the consequence follows.

    Pinned because the first attempt asked each condition for a `readable` that a literal has not got. It passed
    every test, since no test had a consequence with conditions on it, and killed a run on the first capture."""
    when = Clause(
        (Literal("refused", ()), Literal("turn", (Constant("white"),)).denied)
    )

    said = Consequence("Removed", "move", "grid", (), (), None, (when,)).readable

    assert "refused" not in said
    assert "where turn(white)" in said
from openmind.statement.model.drawn import Always, Place
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.structure.model.machine import Machine
from openmind.structure.model.phase import Phase
from openmind.structure.model.record import Record
from openmind.world.model.action import Action
from openmind.statement.model.change import Moved, Placed, Removed, Told
from openmind.world.model.state import State

PLACING, MOVING = Phase("placing", ("place",)), Phase("moving", ("move", "take"))
PHASES = (PLACING, MOVING)


@dataclass(frozen=True, slots=True)
class Cell(Record):
    row: int
    column: int


def a_position(at, **cells):
    """A three by three board with those cells filled, in that phase."""
    held = [[None, None, None], [None, None, None], [None, None, None]]
    for name, value in cells.items():
        row, column = int(name[1]), int(name[2])
        held[row - 1][column - 1] = value
    return State.of(grid=Grid.of(held), phase=Machine(PHASES, at), turn="first")


def placing(row, column, mark):
    """Putting a new thing down: the square is the parameter's, the thing is always the same."""
    return Watched(
        a_position("placing"),
        Action("place", (("cell", Cell(row, column)),)),
        (Placed("grid", (row, column), mark), Told("turn", "second")),
    )


def moving(origin, destination):
    """Carrying a thing from one square to another."""
    return Watched(
        a_position("moving", **{f"c{origin[0]}{origin[1]}": "a thing"}),
        Action("move", (("from", Cell(*origin)), ("to", Cell(*destination)))),
        (Moved("grid", origin, destination), Told("turn", "second")),
    )


def taking(origin, destination):
    """Carrying a thing onto one that was there, which goes first."""
    return Watched(
        a_position("moving", **{f"c{origin[0]}{origin[1]}": "a thing", f"c{destination[0]}{destination[1]}": "another"}),
        Action("take", (("from", Cell(*origin)), ("to", Cell(*destination)))),
        (Removed("grid", destination), Moved("grid", origin, destination), Told("turn", "second")),
    )


def seen():
    return [
        placing(1, 1, "a mark"), placing(2, 3, "a mark"), placing(3, 2, "a mark"),
        moving((1, 1), (1, 2)), moving((2, 2), (3, 3)), moving((3, 1), (1, 3)),
        taking((1, 1), (2, 2)), taking((3, 3), (1, 2)), taking((2, 1), (2, 3)),
    ]


def learned():
    return ConsequenceLearner().learn(seen())


def test_every_action_of_every_phase_is_learned():
    """A game of phases has an action per phase and each does its own thing. What is learned is what *this*
    action leads to, never what actions lead to, so every action offered anywhere has to come back."""
    found = {one.action for one in learned()}

    assert found == {action for phase in PHASES for action in phase.actions}


def test_what_an_action_does_is_said_in_terms_of_the_action():
    """The whole point. A change naming the square a thing was carried from says what happened once; the same
    change with that square drawn from the action says what happens whenever it is played."""
    carried = next(one for one in learned() if one.action == "move" and one.change == "Moved")

    assert carried.where == (Place("from", "row"), Place("from", "column"))
    assert carried.onto == (Place("to", "row"), Place("to", "column"))


def test_a_move_is_about_the_thing_where_the_move_starts():
    """What ties an action to its subject. Once it is known that a move carries whatever stands where `from`
    points, the thing being moved is not one of nine things the position holds that a rule might be about — it is
    the thing the change names."""
    carried = next(one for one in learned() if one.action == "move" and one.change == "Moved")

    assert {one.parameter for one in carried.where} == {"from"}


def test_a_thing_put_down_that_is_always_the_same_is_said_to_be_always_the_same():
    """Where a game's action names what to place, the thing is drawn from the action; where it does not, it is a
    fact about the action rather than about this playing of it."""
    put = next(one for one in learned() if one.action == "place" and one.change == "Placed")

    assert put.value == Always("a mark")
    assert put.where == (Place("cell", "row"), Place("cell", "column"))


def test_taking_removes_where_the_move_lands_and_not_where_it_starts():
    """Two changes of different kinds in one action, each drawn its own way — which is what tells a move that
    takes from a move onto an empty square, and cannot be had from the two positions."""
    taken = next(one for one in learned() if one.action == "take" and one.change == "Removed")

    assert taken.where == (Place("to", "row"), Place("to", "column"))


def test_an_action_that_does_a_thing_twice_is_left_unlearned_rather_than_learned_wrongly():
    """Castling moves two pieces, a deal gives a card to each player. Which sighting's first is which sighting's
    second is not something agreement alone can settle, so nothing is said."""
    twice = Watched(
        a_position("moving"),
        Action("castle", (("side", "king"),)),
        (Moved("grid", (1, 1), (1, 2)), Moved("grid", (1, 3), (1, 4))),
    )

    assert not [one for one in ConsequenceLearner().learn([twice, twice]) if one.change == "Moved"]


def paying(row, column, won):
    """A move that ends the game, paying each player under their own name."""
    where = State.of(
        grid=Grid.of([[None, None, None], [None, None, None], [None, None, None]]),
        phase=Machine(PHASES, "moving"),
        turn="first",
        payoff=Map.of({"first": None, "second": None}),
    )
    return Watched(
        where,
        Action("place", (("cell", Cell(row, column)),)),
        (
            Placed("grid", (row, column), "a mark"),
            Placed("payoff", ("first",), 1.0 if won == "first" else 0.0),
            Placed("payoff", ("second",), 1.0 if won == "second" else 0.0),
        ),
    )


def test_what_a_game_pays_each_player_is_learned_although_it_happens_once_per_player():
    """Two changes of a kind are left unlearned because nothing tells them apart, and that is right for a board:
    which sighting's first move goes with which sighting's second cannot be had from agreement.

    A map's entries tell themselves apart. What a game pays is one change per player, under that player's own
    name, and the names are the same in every sighting — so there is nothing to pair, and a game being won
    became learnable the moment they were grouped by name instead of being thrown away together."""
    learned = ConsequenceLearner().learn(
        [paying(1, 1, "first"), paying(2, 2, "second"), paying(3, 3, "first"), paying(1, 3, "second")]
    )

    paid = [one for one in learned if one.model == "payoff"]
    assert len(paid) == 2
    assert all(one.change == "Placed" for one in paid)


def test_two_moves_of_one_board_are_still_left_unlearned():
    """The rule this narrows is still the rule everywhere it was right: a grid's coordinates move with the
    action, so keying on them would make every sighting a group of its own and nothing would generalise."""
    twice = Watched(
        a_position("moving"),
        Action("castle", (("side", "king"),)),
        (Moved("grid", (1, 1), (1, 2)), Moved("grid", (1, 3), (1, 4))),
    )

    assert not [one for one in ConsequenceLearner().learn([twice, twice]) if one.change == "Moved"]
