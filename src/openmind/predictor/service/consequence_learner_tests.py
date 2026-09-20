from openmind.predictor.model.drawn import Column, Other, Row
from openmind.predictor.service.consequence_learner import ConsequenceLearner
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.world.model.action import Action
from openmind.world.model.state import State
from openmind.world.service.changer import Changer

NAMES = CellNames(("a", "b", "c"), ("3", "2", "1"))
PLAYERS = ("white", "black")


def a_board(pieces, colors, turn="white"):
    return State.of(piece=Grid.of(pieces, NAMES), color=Grid.of(colors, NAMES), turn=turn)


def nothing():
    return [[None, None, None], [None, None, None], [None, None, None]]


def walking(source, target):
    return Action("move", (("source", source), ("target", target)))


def watched(before, action, after, acting="white"):
    return (before, action, acting, Changer().between(before, after), {"piece at source": before.model("piece").at(dict(action.parameters)["source"])})


def test_what_an_action_always_does_is_learned_without_conditions():
    pieces, colors = nothing(), nothing()
    pieces[2][0], colors[2][0] = "rook", "white"
    before = a_board(pieces, colors)
    moved, moved_colors = nothing(), nothing()
    moved[2][2], moved_colors[2][2] = "rook", "white"
    after = a_board(moved, moved_colors, "black")

    found = ConsequenceLearner().learn([watched(before, walking("a1", "c1"), after)], PLAYERS, positions=1)
    said = [one.readable for one in found]

    assert any("moved piece" in one and "the row of source" in one and "the column of target" in one for one in said), said
    assert any("told turn, holding the player not acting" == one for one in said), said


def test_a_square_no_parameter_names_is_learned_as_rows_and_columns_of_the_ones_that_do():
    """The piece taken in passing: at the row the mover started on and the column it landed on."""
    pieces, colors = nothing(), nothing()
    pieces[1][0], colors[1][0] = "pawn", "white"
    pieces[1][1], colors[1][1] = "pawn", "black"
    before = a_board(pieces, colors)
    moved, moved_colors = nothing(), nothing()
    moved[0][1], moved_colors[0][1] = "pawn", "white"
    after = a_board(moved, moved_colors, "black")

    found = ConsequenceLearner().learn([watched(before, walking("a2", "b3"), after)], PLAYERS, positions=1)
    removals = [one for one in found if one.change == "Removed" and one.model == "piece"]

    assert any(one.where == (Row("source"), Column("target")) for one in removals), [one.readable for one in removals]


def test_what_happens_only_sometimes_is_learned_with_the_conditions_it_happens_under():
    quiet_before = a_board(*_with({(2, 0): ("rook", "white")}))
    quiet_after = a_board(*_with({(2, 2): ("rook", "white")}), "black")
    taking_before = a_board(*_with({(2, 0): ("rook", "white"), (2, 2): ("pawn", "black")}))
    taking_after = a_board(*_with({(2, 2): ("rook", "white")}), "black")
    seen = [
        watched(quiet_before, walking("a1", "c1"), quiet_after),
        watched(taking_before, walking("a1", "c1"), taking_after),
    ]

    found = ConsequenceLearner().learn(seen, PLAYERS, positions=1, grow=1.0)
    removals = [one for one in found if one.change == "Removed"]

    assert removals, [one.readable for one in found]
    assert all(one.when for one in removals), [one.readable for one in removals]


def _with(held):
    pieces, colors = nothing(), nothing()
    for (row, column), (piece, color) in held.items():
        pieces[row][column], colors[row][column] = piece, color
    return pieces, colors
