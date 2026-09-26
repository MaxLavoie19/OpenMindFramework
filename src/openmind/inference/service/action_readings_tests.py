from openmind.inference.model.sides import Sides
from openmind.inference.service.action_readings import (
    ACTING_OWNS,
    ANOTHER_OWNS,
    AT,
    BELONGS,
    DISTANCE,
    REACHED,
    REACHED_HOLDING,
    REACHED_OWNED,
    REACHES,
    ROWS,
    ActionReadings,
    Reading,
    a_reading,
    afterwards,
)
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant
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


def test_a_reading_of_one_place_becomes_a_literal_of_that_place_and_what_was_read() -> None:
    found = a_reading(AT, "walker", model="piece", parameter="source").literal

    assert found == Literal("at", (Constant("piece"), Constant("source"), Constant("walker")))


def test_a_reading_of_two_places_keeps_both_of_them_apart() -> None:
    found = a_reading(ROWS, 3, first="source", second="target").literal

    assert found == Literal("rows", (Constant("source"), Constant("target"), Constant(3)))


def test_the_same_question_asked_of_different_places_is_one_predicate_and_not_two() -> None:
    """Which is the whole of what a literal buys over a name. Asked of two pairs of places it is one predicate
    with different terms, so a clause may put a variable where a place is and mean both."""
    found = [
        a_reading(ROWS, 1, first="source", second="target").literal,
        a_reading(ROWS, -1, first="target", second="source").literal,
    ]

    assert {one.predicate for one in found} == {"rows"}


def test_a_reading_about_a_player_keeps_the_player_as_a_term_so_a_rule_can_quantify_over_it() -> None:
    found = a_reading(REACHES, True, player="first", parameter="target").literal

    assert found.arguments[0] == Constant("first")


def test_the_four_reaches_are_four_predicates_although_they_are_written_alike() -> None:
    """`{player} can reach {parameter}`, `{player} can reach {holding}`, `{player} can reach {model} {value!r}`
    and `{player} can reach {whose} {holding}` all come out as some text, " can reach ", and more text — so a
    filled name cannot say which it came from, and matching one against the templates in order put them all
    under whichever was listed first.

    It is not a nicety. `another player can reach source` is the condition the king-safety rule is made of, and
    flattened it came back as `can reach holding('another player', 'source', True)` — naming a square the action
    points at as though it were a thing standing on one. A clause tying that argument to another reading's
    holding would have tied a square to a piece."""
    found = [
        a_reading(REACHES, True, player="first", parameter="source").literal,
        a_reading(REACHED_HOLDING, True, player="first", holding="color 'white' and piece 'king'").literal,
        a_reading(REACHED, True, player="first", model="piece", value="king").literal,
        a_reading(REACHED_OWNED, True, player="first", whose="another player's", holding="piece 'king'").literal,
    ]

    assert len({one.predicate for one in found}) == 4
    assert [len(one.arguments) for one in found] == [3, 3, 4, 4]
    assert found[0] == Literal("can reach", (Constant("first"), Constant("source"), Constant(True)))


def test_a_role_with_a_space_in_it_stays_one_term() -> None:
    """`another player's` is one thing and was two: split at the space it left `another` as the role and
    `player's piece 'king'` as what stands there."""
    found = a_reading(REACHED_OWNED, True, player="first", whose=ANOTHER_OWNS, holding="piece 'king'").literal

    assert found.arguments[1] == Constant("another player's")
    assert found.arguments[2] == Constant("piece 'king'")


def test_whose_a_thing_is_keeps_the_side_apart_from_the_place() -> None:
    found = a_reading(BELONGS, True, model="piece", parameter="source", whose=ACTING_OWNS).literal

    assert found.predicate == "belongs"
    assert found.arguments[:3] == (Constant("piece"), Constant("source"), Constant("the one acting's"))


def test_a_reading_of_the_position_the_action_leads_to_says_so_and_keeps_its_own_shape() -> None:
    found = afterwards(a_reading(AT, "walker", model="piece", parameter="target")).literal

    assert found.predicate == "after it, at"
    assert found.arguments == (Constant("piece"), Constant("target"), Constant("walker"))


def test_a_reading_no_template_accounts_for_is_kept_whole_rather_than_thrown_away() -> None:
    """A game says things in its own way — a scalar of its own, a parameter of its own — and an unknown reading
    is still evidence."""
    found = Reading("something a game named its own way", value=4).literal

    assert found == Literal("something a game named its own way", (Constant(4),))


def test_what_was_read_keeps_the_type_it_was_read_as() -> None:
    """Nobody parses it back out of a name, so nobody can guess it wrong. A game whose values are the characters
    `1` to `8` keeps them as characters, where reading them out of a name made every one of them a number."""
    found = a_reading(AT, "3", model="piece", parameter="source").literal

    assert found.arguments[2] == Constant("3")
    assert found.arguments[2] != Constant(3)


def test_readings_gathered_with_their_labels_become_cases_to_learn_from() -> None:
    seen = [
        ((a_reading(AT, "walker", model="piece", parameter="source"),), True),
        ((a_reading(AT, "flyer", model="piece", parameter="source"),), False),
    ]

    found = ActionReadings().examples(seen, ("a position", "another position"))

    assert [one.holds for one in found] == [True, False]
    assert [one.where for one in found] == ["a position", "another position"]


def test_every_reading_of_a_case_becomes_a_literal_of_that_case() -> None:
    readings = (
        a_reading(AT, "walker", model="piece", parameter="source"),
        a_reading(DISTANCE, 1, first="source", second="target"),
    )

    example = ActionReadings().example(readings, True)

    assert len(example.literals) == 2
    assert {one.predicate for one in example.literals} == {"at", "steps"}
