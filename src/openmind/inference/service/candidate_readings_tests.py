import logging
from dataclasses import dataclass

from openmind.inference.model.evidence import Evidence
from openmind.inference.service.candidate_readings import (
    ACTING,
    ANOTHER,
    BELONGS,
    TOWARD,
    CandidateReadings,
)
from openmind.inference.service.side_deducer import FACES, OWNS
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Functor, Number
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.structure.model.list import List
from openmind.structure.model.map import Map
from openmind.structure.model.record import Record
from openmind.structure.model.value import Value
from openmind.world.model.action import Action
from openmind.world.model.state import State

NAMES = CellNames(("a", "b"), ("2", "1"))


@dataclass(frozen=True, slots=True)
class Piece(Record):
    """What a game of this sort puts on a square, declared by the game and not by OMF."""

    colour: str
    type: str


@dataclass(frozen=True, slots=True)
class Square(Record):
    """A square has a colour of its own and whatever stands on it. It has no coordinates: where it sits is the
    grid's business, and asking a square where it is would be asking it something it does not know."""

    colour: str
    piece: Value = None


def a_position():
    """Two columns and two rows, named the way a game names cells: row 1 is the top, so row 1 column 1 is a2."""
    return State.of(
        grid=Grid.of(
            [
                [Square("white", Piece("white", "rook")), Square("black")],
                [Square("black"), Square("white", Piece("black", "king"))],
            ],
            NAMES,
        ),
        turn="white",
    )


def a_piece(colour, type):
    return Functor("piece", (Constant(colour), Constant(type)))


def a_square(colour, piece=Constant(None)):
    """A square as a reading says it: one term, named after what the game called the record, of its parts in order."""
    return Functor("square", (Constant(colour), piece))


def a_move(origin, destination):
    return Action("move", (("destination", destination), ("origin", origin)))


def test_a_grid_cell_is_read_as_its_position_and_what_stands_there():
    readings = CandidateReadings().of_state(a_position())

    assert Literal("grid", (Number(1), Number(1), a_square("white", a_piece("white", "rook")))) in readings


def test_a_record_is_read_as_one_term_of_its_parts_however_deep_they_go():
    """What stands in a cell is one thing and is said as one thing, and a thing holding a thing is a term holding a
    term. Nothing is lost by it: a clause puts a variable wherever it does not care, and unification goes inside a
    term, so asking whose the piece on a square is and nothing else reads
    `grid(Row, Column, square(Anything, piece(white, Something)))`."""
    readings = CandidateReadings().of_state(a_position())

    assert Literal("grid", (Number(2), Number(2), a_square("white", a_piece("black", "king")))) in readings


def test_a_square_has_a_colour_of_its_own_and_no_coordinates():
    """Its position in the grid is its coordinates. A square is not asked for them and does not carry them, so a
    reading says where it is once — in the places naming the cell — and never again inside the term."""
    readings = CandidateReadings().of_state(a_position())

    assert Literal("grid", (Number(1), Number(2), a_square("black"))) in readings


def test_an_empty_square_is_read_as_holding_nothing_rather_than_left_out():
    """A square nothing stands on is why a move is possible at all, so it has to be sayable. Left out, a rule could
    never ask for it and every rule about moving to an empty square would be out of reach."""
    readings = CandidateReadings().of_state(a_position())

    assert Literal("grid", (Number(2), Number(1), a_square("black", Constant(None)))) in readings


def test_a_scalar_a_map_and_a_list_are_read_under_their_own_names():
    state = State.of(turn="white", payoff=Map.of({"white": 1, "black": -1}), played=List(("a2", "b1")))

    readings = CandidateReadings().of_state(state)

    assert Literal("turn", (Constant("white"),)) in readings
    assert Literal("payoff", (Constant("white"), Constant(1))) in readings
    assert Literal("played", (Number(1), Constant("a2"))) in readings


def test_what_stands_in_a_cell_is_a_thing_even_where_it_is_spelled_with_a_digit():
    """Coordinates are numbers so that arithmetic reaches them and one can be tied to another. What stands in a
    cell is not, whatever it looks like: a game that spells a white knight 1 and a black rook 2 has named two
    pieces, and letting the learner conclude that one of them is less than the other would be letting it conclude
    nonsense."""
    readings = CandidateReadings().of_state(State.of(grid=Grid.of([[1, 2]])))

    assert Literal("grid", (Number(1), Number(1), Constant(1))) in readings
    assert Literal("grid", (Number(1), Number(2), Constant(2))) in readings


def test_a_piece_s_colour_travels_with_it_when_it_moves():
    """This is what a record buys over parts kept side by side. Colour and kind held in two grids can be moved out
    of step by a bug, and then a square holds a white king's colour and a black rook's kind, which is no piece at
    all. One value cannot come apart from itself.

    The square's own colour stays where it is, because it is the square's and not the piece's."""
    position = a_position()
    grid = position.model("grid")
    moved = position.with_model(
        "grid",
        grid.placed("a2", Square("white")).placed("b2", Square("black", Piece("white", "rook"))),
    )

    readings = CandidateReadings().of_state(moved)

    assert Literal("grid", (Number(1), Number(2), a_square("black", a_piece("white", "rook")))) in readings
    assert Literal("grid", (Number(1), Number(1), a_square("white"))) in readings


def test_a_parameter_naming_a_cell_is_read_as_that_cell_s_coordinates():
    """Said as its name, a1 and a6 are two unrelated things and nothing could ever relate them. Said as
    coordinates, what they share is there for a clause to find."""
    readings = CandidateReadings().read(a_position(), a_move("a2", "b1"))

    assert Literal("origin", (Number(1), Number(1))) in readings
    assert Literal("destination", (Number(2), Number(2))) in readings


def test_a_parameter_that_names_no_cell_is_read_as_what_it_is():
    state = State.of(pot=0)
    readings = CandidateReadings().read(state, Action("bet", (("amount", 5), ("bluffing", True))))

    assert Literal("amount", (Number(5),)) in readings
    assert Literal("bluffing", (Constant(True),)) in readings


def test_nothing_is_read_but_the_state_s_own_models_and_the_candidate_s_parameters():
    """The vocabulary is the whole of what any learned rule will ever be able to say, so it is pinned here. A grid
    also offers its diagonals, its lines, its rays, its neighbours and the distance between two cells, and every
    one of those is a relation the learner is supposed to discover rather than be handed."""
    readings = CandidateReadings().read(a_position(), a_move("a2", "b1"))

    assert {one.predicate for one in readings} == {
        "grid", "turn", "origin", "destination", "places apart", "holds", "from nothing", "lands on",
        "on line",
    }


def test_what_lies_between_is_read_only_where_there_is_a_line_to_lie_on():
    """"The way is clear" is a statement about every cell between two others, which a clause with one conclusion
    cannot make. "Something is in the way" is a statement that some cell between them holds a thing, and that is
    what a refusal needs — so one reading per cell in between, and none at all where the two do not lie on a
    line, since then nothing is passed over."""
    readings = CandidateReadings().read(a_position(), a_move("a2", "b1"))

    between = [one for one in readings if one.predicate == "on line"]

    assert all(one.arguments[-2].name in ("grid",) for one in between)


def test_where_a_move_said_as_a_step_lands_is_read_although_no_parameter_names_it():
    """A move said as how far it goes names its mover and its step, and the square it reaches is neither. Without
    this, you may not land on your own thing is out of reach — the most basic rule there is."""
    readings = CandidateReadings().read(a_position(), a_move("a2", "b1"))

    landing = [one for one in readings if one.predicate == "lands on"]

    assert landing
    assert all(one.arguments[-2].name == "grid" for one in landing)


def test_a_step_that_leaves_the_grid_says_so_rather_than_saying_nothing():
    """You may not move off it is a rule like the others, and a rule cannot be built out of a reading that is
    simply missing."""
    readings = CandidateReadings().read(a_position(), a_move("a2", "b1"))

    assert any(
        one.predicate == "lands on" and one.arguments[-1] == Constant("outside") for one in readings
    )


def test_two_sizes_are_compared_so_a_rule_about_them_is_in_the_space_at_all():
    """A constraint is built out of the readings a case carried and out of nothing else — generalising drops a
    reading or puts a variable in it, and never adds one. So a comparison that is not read is a comparison no
    clause can ever mention, and a thing travelling as far one way as the other is exactly such a comparison."""
    readings = CandidateReadings().read(a_position(), a_move("a2", "b1"))

    sizes = [
        one
        for one in readings
        if one.predicate == "places apart" and str(one.arguments[0].name).startswith("how far")
    ]

    assert sizes
    assert all(one.arguments[-1].value >= 0 for one in sizes)


def test_sizes_that_are_equal_are_read_as_standing_alongside_each_other():
    """Which is the whole of the question for anything moving as far one way as the other, and it holds however
    the two are signed — the size is read apart from which side of nothing it falls on."""
    readings = CandidateReadings().read(a_position(), a_move("a2", "b1"))

    said = {
        (one.arguments[0].name, one.arguments[1].name): one.arguments[2].name
        for one in readings
        if one.predicate == "places apart" and str(one.arguments[0].name).startswith("how far")
    }
    sized = {
        one.arguments[0].name: one.arguments[-1].value
        for one in readings
        if one.predicate == "from nothing"
    }

    for (one, other), way in said.items():
        equal = sized[one.removeprefix("how far ")] == sized[other.removeprefix("how far ")]
        assert (way == "alongside") == equal


def test_how_big_a_number_is_is_read_apart_from_which_side_of_nothing_it_is_on():
    """Two numbers being the same size is not two numbers being apart by nothing. A thing moving by as much one
    way as the other does it along four diagonals, and the distance between the two offsets is nothing for two of
    them and twice the distance for the other two — so the one rule cannot be said and half the directions get no
    rule at all. Said from nothing, all four carry the same distance in both places."""
    readings = CandidateReadings().read(a_position(), a_move("a2", "b1"))

    said = {one.arguments[0].name: one.arguments[1:] for one in readings if one.predicate == "from nothing"}
    pointed = {
        one.predicate: tuple(term.value for term in one.arguments)
        for one in readings
        if one.predicate in ("origin", "destination")
    }

    assert set(said) == {"origin 1", "origin 2", "destination 1", "destination 2"}
    assert all(way.name == "after" and far.value >= 0 for way, far in said.values())
    assert said["origin 1"][1].value == pointed["origin"][0]


def test_a_case_holds_where_the_game_refuses_the_candidate():
    """Covering a case is refusing it. Being legal is no clause covering the case and nothing else, which is what
    having no generators comes to in practice."""
    evidence = Evidence(a_position(), "move", (a_move("a2", "a1"),))

    cases = CandidateReadings().cases(evidence, {"origin": ("a2", "a1"), "destination": ("a2", "a1")})

    allowed = [one for one in cases if not one.holds]
    assert len(allowed) == 1
    assert Literal("origin", (Number(1), Number(1))) in allowed[0].literals
    assert Literal("destination", (Number(2), Number(1))) in allowed[0].literals


def test_every_assignment_the_domains_allow_becomes_a_case():
    """Nothing is sampled and nothing is left out: the cases are the whole space the solver will later search, so
    a constraint that accounts for all of them accounts for everything the solver can propose."""
    evidence = Evidence(a_position(), "move", ())

    cases = CandidateReadings().cases(evidence, {"origin": ("a2", "a1", "b2"), "destination": ("a2", "a1")})

    assert len(cases) == 6
    assert all(one.holds for one in cases)


def test_a_case_says_which_position_it_came_from():
    """Two cases from one position are not two pieces of evidence the way two from two positions are, so what a
    clause is later judged on has to know where each case came from."""
    position = a_position()
    cases = CandidateReadings().cases(Evidence(position, "move", ()), {"origin": ("a2",), "destination": ("a1",)})

    assert cases[0].where == position


def test_a_legal_action_no_domain_could_propose_is_said_out_loud(caplog):
    """No constraint can ever account for a move that was never a candidate. Left unsaid, the learner goes on
    failing to explain it for ever with nothing to say why."""
    evidence = Evidence(a_position(), "move", (a_move("b1", "b2"),))

    with caplog.at_level(logging.WARNING):
        CandidateReadings().cases(evidence, {"origin": ("a2",), "destination": ("a1",)})

    assert "no domain could propose" in caplog.text


def test_the_position_is_read_once_and_shared_by_every_candidate():
    """A position says the same of itself whatever move is considered in it. Reading it again per candidate is the
    difference between reading a position and reading it four thousand times."""
    readings = CandidateReadings()
    evidence = Evidence(a_position(), "move", ())

    cases = readings.cases(evidence, {"origin": ("a2", "a1"), "destination": ("a2", "a1")})

    shared = readings.of_state(evidence.where)
    assert all(one.literals[: len(shared)] == shared for one in cases)


def test_a_parameter_of_parts_is_read_with_one_place_per_part():
    """This version of chess says `move(origin(row, column), destination(row, column))`, so a parameter pointing
    at a cell arrives as two numbers rather than as a name. They are numbers, not things: what a parameter ranges
    over is declared, and arithmetic has to reach it."""

    @dataclass(frozen=True, slots=True)
    class Cell(Record):
        row: int
        column: int

    move = Action("move", (("destination", Cell(4, 5)), ("origin", Cell(4, 1))))

    readings = CandidateReadings().read(a_position(), move)

    assert Literal("origin", (Number(4), Number(1))) in readings
    assert Literal("destination", (Number(4), Number(5))) in readings


def a_side_deduction():
    """What the deduction found about this game: whose the things are, and which way each player faces."""
    return (
        Literal(OWNS, (Constant("white"), Constant("at grid"), a_square("white", a_piece("white", "rook")))),
        Literal(OWNS, (Constant("black"), Constant("at grid"), a_square("white", a_piece("black", "king")))),
        Literal(FACES, (Constant("white"), Number(1))),
        Literal(FACES, (Constant("black"), Number(-1))),
    )


def reading_with_sides():
    return CandidateReadings(sides=a_side_deduction(), acting=lambda state: state.value("turn"))


def test_whose_a_thing_is_is_read_as_a_role_so_one_rule_serves_both_players():
    """Written over colours, "you may not move another player's thing" is two rules that each name a side and
    neither of which says what it means. Read as a role it is one."""
    found = reading_with_sides().read(a_position(), a_move("a2", "b2"))

    said = {one for one in found if one.predicate == BELONGS}
    assert said
    assert {one.arguments[-1].name for one in said} <= {ACTING, ANOTHER}


def test_a_game_that_deduced_no_sides_is_offered_nothing_about_them():
    """A game whose things belong to nobody, or whose pieces all move both ways, has no side to tell — and then
    the vocabulary is what it was, rather than carrying a reading that always says the same thing."""
    found = CandidateReadings().read(a_position(), a_move("a2", "b2"))

    assert not [one for one in found if one.predicate in (BELONGS, TOWARD)]


def test_a_game_whose_things_are_owned_but_go_both_ways_is_told_whose_and_not_which_way():
    """Stratego: every piece belongs to a side and none of them has a forward, so the deduction finds owners
    and no facing. Whose a thing is still has to be sayable, and how far a move goes toward anything must not
    be — a number invented for an axis the game has not got would be read as a rule about one."""
    owners = tuple(one for one in a_side_deduction() if one.predicate == OWNS)
    reading = CandidateReadings(sides=owners, acting=lambda state: state.value("turn"))

    found = reading.read(a_position(), a_move("a2", "b2"))

    assert [one for one in found if one.predicate == BELONGS]
    assert not [one for one in found if one.predicate == TOWARD]


def test_how_far_a_move_goes_is_read_the_way_its_own_player_faces():
    """A pawn's step is one row up for one player and one row down for the other, so every pawn rule was learned
    twice and carried a colour inside a rule that is not about colour. Read toward the player's own side, the
    two are one."""
    reading = reading_with_sides()
    white = {one.arguments[-1].value for one in reading.read(a_position(), a_move("a2", "b2"))
             if one.predicate == TOWARD}
    black = {one.arguments[-1].value for one in
             reading.read(a_position().with_model("turn", "black"), a_move("a2", "b2"))
             if one.predicate == TOWARD}

    assert white and black
    assert white == {-one for one in black}


def test_what_a_game_has_paid_is_left_out_of_the_readings_where_the_caller_says_so():
    """Legality is what produces a result, so a rule conditioned on the result is backwards: a game is not over
    because somebody won, somebody won because the game was over.

    Worse where a game pays out precisely when nothing was legal — a constraint reading the payoff would restate
    the observation, correctly and cheaply, while teaching nothing about how anything moves. The position keeps
    carrying it, because that is how a predictor sees a game being won."""
    from openmind.structure.model.map import Map

    over = State.of(
        grid=Grid.of([["a thing", None], [None, None]]),
        turn="white",
        payoff=Map.of({"white": 1.0, "black": 0.0}),
    )
    action = Action("move", (("origin", "a2"), ("destination", "b2")))

    read = CandidateReadings().read(over, action)
    left_out = CandidateReadings(paid=("payoff",)).read(over, action)

    assert "payoff" in {one.predicate for one in read}
    assert "payoff" not in {one.predicate for one in left_out}
    assert "turn" in {one.predicate for one in left_out}
