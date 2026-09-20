from openmind.inference.model.sides import Sides
from openmind.inference.service.action_readings import ActionReadings
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.world.model.action import Action
from openmind.world.model.state import State

NAMES = CellNames(("a", "b"), ("2", "1"))


class TwoSquaresReaching:
    """A stand-in for a game's reach: each player's moves land on the squares named for them."""

    def __init__(self, landing):
        self._landing = landing

    def players(self, state):
        return tuple(self._landing)

    def cells(self, state, player):
        return tuple(NAMES.to_coordinates(one) for one in self._landing.get(player, ()))


def a_board(landing):
    state = State.of(
        piece=Grid.of([["king", "pawn"], [None, "rook"]], NAMES),
        color=Grid.of([["white", "black"], [None, "white"]], NAMES),
        turn="white",
    )
    action = Action("move", (("source", "b1"), ("target", "a1")))
    sides = Sides((("color", "white", "white"), ("color", "black", "black")), (("white", 1), ("black", -1)))
    return state, action, sides, TwoSquaresReaching(landing)


def test_a_piece_of_your_own_is_attacked_where_another_player_can_reach_its_square():
    state, action, sides, reach = a_board({"black": ("b1",), "white": ()})

    readings = ActionReadings().of(state, action, None, reach, "white", sides)

    assert readings["another player can reach source"] is True
    assert readings["the one acting can reach source"] is False


def test_a_square_nobody_can_reach_is_attacked_by_nobody():
    state, action, sides, reach = a_board({"black": ("a2",), "white": ()})

    readings = ActionReadings().of(state, action, None, reach, "white", sides)

    assert readings["another player can reach source"] is False


def test_what_a_player_can_reach_is_read_by_whose_it_is():
    """King safety said once: the others can reach the acting player's king, whatever colour that is today."""
    state, action, sides, reach = a_board({"black": ("a2",), "white": ()})

    readings = ActionReadings().of(state, action, None, reach, "white", sides)

    assert readings["another player can reach the one acting's piece 'king'"] is True
    assert readings["another player can reach another player's piece 'pawn'"] is False


def test_whose_a_square_is_is_read_after_the_action_as_well_as_before():
    """What was theirs and is yours afterwards is a capture, seen rather than named."""
    state, _, sides, reach = a_board({"black": (), "white": ()})
    taking = Action("move", (("source", "a2"), ("target", "b2")))
    taken = State.of(
        piece=Grid.of([[None, "king"], [None, "rook"]], NAMES),
        color=Grid.of([[None, "white"], [None, "white"]], NAMES),
        turn="black",
    )

    readings = ActionReadings().of(state, taking, taken, reach, "white", sides)

    assert readings["color at target is another player"] is True
    assert readings["after it, color at target is the one acting"] is True
    assert readings["after it, color at target is another player"] is False


def test_reach_is_read_of_the_position_as_well_as_of_the_one_it_leads_to():
    state, action, sides, reach = a_board({"black": ("b1",), "white": ()})

    readings = ActionReadings().of(state, action, state, reach, "white", sides)

    assert "another player can reach source" in readings
    assert "after it, another player can reach source" in readings
