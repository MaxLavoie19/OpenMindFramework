import logging
import tempfile
from pathlib import Path
from typing import NamedTuple

import pytest

from openmind.inference.service.fact_recorder import FactRecorder
from openmind.inference.service.side_deducer import FACES, OWNS, SideDeducer
from openmind.knowledge.factory.knowledge_base_factory import create_knowledge_base
from openmind.statement.model.literal import Literal
from openmind.statement.model.term import Constant, Number


class Thing(NamedTuple):
    """A thing a game spells as a record of parts, as the constraint learner's chess grid does."""

    colour: str
    kind: str


def moving(piece, color, rows, columns=0, at="b2"):
    """One allowed action, read as the literals a vocabulary says it in.

    **The parameters are read under their own names, because that is what a vocabulary does.** An earlier
    version of this left them out, and every test here passed while the deduction found nothing on a real
    game — a parameter gives itself away by being read on its own and named by other readings, so a fixture
    without them cannot exercise the thing that finds them."""
    return (
        Literal("source", (Constant(at),)),
        Literal("target", (Constant(f"{at[0]}{int(at[1]) + rows}"),)),
        Literal("at", (Constant("piece"), Constant("source"), Constant(piece))),
        Literal("at", (Constant("color"), Constant("source"), Constant(color))),
        Literal("rows", (Constant("source"), Constant("target"), Number(rows))),
        Literal("columns", (Constant("source"), Constant("target"), Number(columns))),
    )


def a_game():
    """Legal actions of two players, where pawns only ever advance and rooks go both ways.

    A pawn takes to either side, so the columns it moves through go both ways while its rows go one way. That
    is what makes the rows the answer and not merely the first thing tried: with a pawn that only ever took to
    the right, the columns would be as one-way as the rows and the evidence would not tell them apart."""
    return [
        (moving("pawn", "white", 1), "white"),
        (moving("pawn", "white", 1, 1), "white"),
        (moving("pawn", "white", 1, -1), "white"),
        (moving("rook", "white", 3), "white"),
        (moving("rook", "white", -2), "white"),
        (moving("pawn", "black", -1), "black"),
        (moving("pawn", "black", -1, -1), "black"),
        (moving("pawn", "black", -1, 1), "black"),
        (moving("rook", "black", -4), "black"),
        (moving("rook", "black", 2), "black"),
    ]


def in_two_positions(examples):
    """The same actions seen twice, each position holding only the actions of whoever is to move in it.

    **One player moves in a position, which is the whole difference.** Pooling both into every position made
    the mover's colour vary within a position when in a real game it never can, and a filter written against
    that fixture threw away the one reading ownership rests on."""
    found, where = [], []
    for round_number in (0, 1):
        for number, player in enumerate(sorted({one for _, one in examples})):
            for example in [one for one in examples if one[1] == player]:
                found.append(example)
                where.append(round_number * 2 + number)
    return found, where


def test_a_value_only_one_player_ever_moves_belongs_to_that_player():
    examples, positions = in_two_positions(a_game())

    found = SideDeducer().deduce(examples, positions)

    owned = {(one.arguments[1].name, one.arguments[2].name): one.arguments[0].name
             for one in found if one.predicate == OWNS}
    assert ("at piece source", "pawn") not in owned
    assert owned[("at color source", "white")] == "white"
    assert owned[("at color source", "black")] == "black"


def test_a_player_faces_the_way_their_one_way_piece_goes():
    """Said as a fact and not as a field: white moves up the board and black moves down, which is a sentence
    about the game. Nothing about the shape of it is this service's to fix, so a game that wants to say its
    sides some other way is not prevented."""
    examples, positions = in_two_positions(a_game())

    found = SideDeducer().deduce(examples, positions)

    said = {(one.arguments[0].name, one.arguments[-1].value) for one in found if one.predicate == FACES}
    assert ("white", 1) in said
    assert ("black", -1) in said


def test_a_game_whose_pieces_all_move_both_ways_has_no_side_to_tell():
    examples = [
        (moving("rook", "white", 3), "white"),
        (moving("rook", "white", -2), "white"),
        (moving("rook", "black", -4), "black"),
        (moving("rook", "black", 2), "black"),
    ]

    found = SideDeducer().deduce(*in_two_positions(examples))

    assert not [one for one in found if one.predicate == FACES]


def test_a_piece_seen_going_one_way_in_a_single_position_says_nothing():
    """A rook that happened to move up twice is as one-way as a pawn, until another position disagrees."""
    examples = [(moving("rook", "white", 3), "white"), (moving("rook", "white", 2), "white")]

    found = SideDeducer().deduce(examples, [0, 0])

    assert not [one for one in found if one.predicate == FACES]


def test_a_value_both_players_move_belongs_to_neither():
    examples = [
        (moving("boulder", "grey", 1), "white"),
        (moving("boulder", "grey", -1), "black"),
    ]

    found = SideDeducer().deduce(examples, [0, 1])

    assert not [one for one in found if one.arguments[-1] == Constant("boulder")]


def test_a_thing_that_is_a_record_can_be_owned_like_any_other():
    """The old record held `(model, value, player)` and matched a whole bare value, so a grid holding
    `Piece(colour, type)` records — which is what the constraint learner reads — could be asked whose anything
    was and would answer nobody's, silently. A fact says whatever the readings said stood there, so a game whose
    things are records says so."""
    examples = [
        (moving(Thing("white", "pawn"), "white", 1), "white"),
        (moving(Thing("black", "pawn"), "black", -1), "black"),
    ]

    found = SideDeducer().deduce(*in_two_positions(examples))

    owned = {one.arguments[2].name: one.arguments[0].name for one in found if one.predicate == OWNS}
    assert owned[Thing("white", "pawn")] == "white"
    assert owned[Thing("black", "pawn")] == "black"


def test_what_was_deduced_is_still_the_record_the_stand_in_asks_for():
    """Until `ActionReadings` goes, it asks a record a question. That is one presentation of
    the facts and not where they live."""
    examples, positions = in_two_positions(a_game())
    deducer = SideDeducer()

    sides = deducer.sides(deducer.deduce(examples, positions))

    assert sides.whose("color", "white") == "white"
    assert sides.toward("white") == 1
    assert sides.toward("black") == -1


def test_what_was_deduced_is_written_down_where_learned_things_go():
    """A deduction that lives only as long as the run that made it has to be made again every run, and cannot
    be argued with. Written down it is a belief like any other, with what it rests on."""
    examples, positions = in_two_positions(a_game())
    deducer = SideDeducer()

    found = deducer.facts(deducer.deduce(examples, positions))

    with tempfile.TemporaryDirectory() as directory:
        knowledge = create_knowledge_base("chess", Path(directory))
        kept = FactRecorder().record(knowledge, "the rules of chess", found)

    said = {one.variable: one.value for one in kept}
    assert said["faces white rows source target"] == 1
    assert said["faces black rows source target"] == -1
    assert "owns white at color source white" in said


def test_whose_a_thing_is_holds_how_firmly_it_is_held():
    """Whose a thing is used to hold nothing, on the reasoning that it has no quantity about it. True of the
    claim and false of the claiming: a pawn seen four hundred times and a queen of the other colour seen once
    went into the knowledge base alike, and nothing that read them could weigh one against the other."""
    examples, positions = in_two_positions(a_game())
    deducer = SideDeducer()

    found = deducer.facts(deducer.deduce(examples, positions))

    owns = [one for one in found if one.kind == OWNS]
    faces = [one for one in found if one.kind == FACES]
    assert all(isinstance(one.held, float) and 0.0 <= one.held <= 1.0 for one in owns)
    assert all(one.held in (1, -1) for one in faces)


def test_a_reading_that_is_never_negative_says_nothing_about_which_way_anybody_faces():
    """How far apart two places are, how many steps a move takes, how many things stand between — all
    magnitudes, none ever negative, so each is one-way for every player by construction. Taken at face value
    they had both players facing the same way, which is not a thing two players can do on one axis."""
    def also_apart(piece, color, rows, columns=0):
        return (*moving(piece, color, rows, columns), Literal("steps", (Constant("source"), Constant("target"), Number(abs(rows) + abs(columns)))))

    examples = [
        (also_apart("pawn", "white", 1), "white"),
        (also_apart("pawn", "white", 1, 1), "white"),
        (also_apart("pawn", "white", 1, -1), "white"),
        (also_apart("pawn", "black", -1), "black"),
        (also_apart("pawn", "black", -1, -1), "black"),
        (also_apart("pawn", "black", -1, 1), "black"),
    ]
    examples = examples + examples
    deducer = SideDeducer()

    found = deducer.deduce(examples, [0] * 6 + [1] * 6)

    assert not [one for one in found if one.predicate == FACES and one.arguments[1].name.startswith("steps")]
    assert dict(deducer.sides(found).facing) == {"white": 1, "black": -1}


def test_what_was_deduced_is_named_beside_how_firmly_it_is_held(caplog: pytest.LogCaptureFixture):
    """The firm claim and the accident are the same sentence, and only their evidence tells them apart.

    A colour the mover moved every single time is owned about as clearly as this deduces anything. A piece
    kind seen twice, both times because whoever was on the move happened to be the one holding it, satisfies
    "only one player ever moved it" just as well and means almost nothing. Both come out as `owns`, so what
    each was seen on is said with it — and a count on its own, which is what this used to log, could not have
    shown that a real run had made light and dark squares somebody's property."""
    caplog.set_level(logging.INFO, logger="openmind.inference.service.side_deducer")
    examples, positions = in_two_positions([*a_game(), (moving("king", "white", 1), "white")])

    SideDeducer().deduce(examples, positions)

    said = [one for one in caplog.messages if one.startswith("white owns")]
    colour = next(one for one in said if "at color source" in one)
    piece = next(one for one in said if "at piece source" in one)

    assert "white at" in colour and "on 12" in colour
    assert "king at" in piece and "on 2" in piece
    # A claim resting on twelve sightings is held more firmly than one resting on two.
    assert float(colour.split(" at ")[1].split(" on ")[0]) > float(piece.split(" at ")[1].split(" on ")[0])
