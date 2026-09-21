from openmind.inference.model.sides import Sides
from openmind.inference.service.worth_reasoner import WorthReasoner
from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.world.model.action import Action
from openmind.world.model.state import State

NAMES = CellNames(("a", "b", "c"), ("3", "2", "1"))


class AGameOfReach:
    """A game where a thing lets its owner act once for every square it can reach, and a player who can do nothing
    is paid worst. Nothing here is chess; it is the smallest game that has anything to say about worth."""

    def __init__(self, reaches, ended=None):
        self._reaches = reaches
        self._ended = ended or {}

    def players(self):
        return type("Named", (), {"names": ("white", "black")})()

    def actions(self, state, player="white"):
        found = []
        grid = state.model("piece")
        colors = state.model("color")
        for at in grid.coordinates():
            if grid.at(at) is None or colors.at(at) != player:
                continue
            found.extend(Action("move", (("source", grid.alias(at)), ("target", number))) for number in range(self._reaches.get(grid.at(at), 0)))
        return tuple(found)

    def joint_actions(self, state):
        return ((0, self.actions(state, "white")),) if self.actions(state, "white") else ()

    def ended(self, state):
        return self._ended.get(state, None)


def a_board(held):
    pieces = [[None] * 3 for _ in range(3)]
    colors = [[None] * 3 for _ in range(3)]
    for (row, column), (piece, color) in held.items():
        pieces[row][column], colors[row][column] = piece, color
    return State.of(piece=Grid.of(pieces, NAMES), color=Grid.of(colors, NAMES), turn="white")


def test_a_thing_is_worth_what_its_owner_can_no_longer_do_without_it():
    game = AGameOfReach({"queen": 8, "pawn": 1})
    position = a_board({(0, 0): ("queen", "white"), (2, 2): ("pawn", "white")})

    worth = WorthReasoner().reason(game, [position])

    assert worth.of("piece", "queen") == 8
    assert worth.of("piece", "pawn") == 1


def test_only_what_belongs_to_the_one_acting_is_valued_to_them():
    game = AGameOfReach({"queen": 8, "pawn": 1})
    position = a_board({(0, 0): ("queen", "white"), (2, 2): ("pawn", "black")})
    sides = Sides((("color", "white", "white"), ("color", "black", "black")))

    worth = WorthReasoner().reason(game, [position], sides)

    assert worth.of("piece", "queen") == 8
    assert worth.of("piece", "pawn") == 0


def test_the_reasoning_rests_on_the_game_paying_worst_where_nothing_can_be_done():
    empty = a_board({})
    game = AGameOfReach({"queen": 8}, {empty: Map.of({"white": -1.0, "black": 1.0})})
    position = a_board({(0, 0): ("queen", "white")})

    worth = WorthReasoner().reason(game, [position, empty])

    assert worth.ended == 1
    assert worth.settled


def test_a_game_that_pays_worst_where_much_can_be_done_bears_nothing_out():
    """The ground is read, not assumed: a game where the worst-paid player could still act says nothing."""
    busy = a_board({(0, 0): ("queen", "white")})
    game = AGameOfReach({"queen": 8}, {busy: Map.of({"white": -1.0, "black": 1.0})})

    worth = WorthReasoner().reason(game, [busy])

    assert worth.ended == 1
    assert not worth.settled


def test_nothing_is_valued_in_a_position_that_is_over():
    finished = a_board({(0, 0): ("queen", "white")})
    game = AGameOfReach({"queen": 8}, {finished: Map.of({"white": -1.0, "black": 1.0})})

    assert WorthReasoner().reason(game, [finished]).holdings == ()
