import random

from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid
from openmind.world.model.state import State
from openmind.world.service.position_builder import PositionBuilder

NAMES = CellNames(("a", "b"), ("2", "1"))


def a_position() -> State:
    """Two grids on the same cells, the way a game keeps a thing and whose it is, and a scalar beside them."""
    return State.of(
        piece=Grid.of([["rook", "king"], [None, "pawn"]], NAMES),
        colour=Grid.of([["white", "black"], [None, "white"]], NAMES),
        turn="white",
    )


def test_emptying_clears_every_grid_and_leaves_everything_else_alone():
    """A walk from chess's opening never reaches a board with one bishop on it, so a position is built rather
    than reached — and what is not a grid is not the builder's business."""
    emptied = PositionBuilder().emptied(a_position())

    assert all(emptied.model("piece").at(one) is None for one in emptied.model("piece").coordinates())
    assert all(emptied.model("colour").at(one) is None for one in emptied.model("colour").coordinates())
    assert emptied.model("turn").value == "white"


def test_emptying_keeps_the_shape_and_the_names_a_game_reads_cells_by():
    """An emptied board a game cannot name its cells on is not that game's board."""
    before = a_position()

    emptied = PositionBuilder().emptied(before)

    assert emptied.model("piece").shape == before.model("piece").shape
    assert emptied.model("piece").at("a2") is None, "still nameable as the game names it"


def test_placing_sets_the_cells_it_is_given_and_no_others():
    placed = PositionBuilder().placed(
        PositionBuilder().emptied(a_position()), {"piece": {"b1": "queen"}, "colour": {"b1": "black"}}
    )

    assert placed.model("piece").at("b1") == "queen"
    assert placed.model("colour").at("b1") == "black"
    assert placed.model("piece").at("a2") is None


def test_what_is_placed_does_not_have_to_be_a_position_the_game_could_reach():
    """An impossible position still answers what the rules allow in it, which is what is being found out."""
    placed = PositionBuilder().placed(a_position(), {"piece": {"a2": "king", "b2": "king", "a1": "king"}})

    assert [placed.model("piece").at(one) for one in ("a2", "b2", "a1")] == ["king", "king", "king"]


def test_scattering_fills_the_same_cells_in_every_grid():
    """What a game keeps in one grid stays with what it keeps in another: a piece and whose it is. A piece with
    no colour, or a colour with no piece, is a board no rule can be tested on."""
    scattered = PositionBuilder().scattered(
        a_position(), {"piece": ["rook", "pawn"], "colour": ["white", "black"]}, pieces=2, rng=random.Random(1)
    )

    pieces = {one for one in scattered.model("piece").coordinates() if scattered.model("piece").at(one)}
    colours = {one for one in scattered.model("colour").coordinates() if scattered.model("colour").at(one)}
    assert pieces == colours
    assert len(pieces) == 2


def test_scattering_takes_each_grid_s_value_from_what_that_grid_was_given():
    scattered = PositionBuilder().scattered(
        a_position(), {"piece": ["rook"], "colour": ["black"]}, pieces=3, rng=random.Random(2)
    )

    held = [scattered.model("piece").at(one) for one in scattered.model("piece").coordinates()]
    whose = [scattered.model("colour").at(one) for one in scattered.model("colour").coordinates()]
    assert set(held) == {"rook", None}
    assert set(whose) == {"black", None}


def test_asking_for_more_cells_than_there_are_fills_the_board_rather_than_raising():
    scattered = PositionBuilder().scattered(
        a_position(), {"piece": ["rook"], "colour": ["white"]}, pieces=99, rng=random.Random(3)
    )

    assert all(scattered.model("piece").at(one) == "rook" for one in scattered.model("piece").coordinates())


def test_a_scalar_left_alone_is_how_a_built_position_stays_like_the_one_it_came_from():
    """Every board with the same player to move, and a rule that only held because of that survives a test it
    should have failed. Said as a test because it is the trap, not a feature."""
    builder, values = PositionBuilder(), {"piece": ["rook"], "colour": ["white"]}

    kept = builder.scattered(a_position(), values, pieces=1, rng=random.Random(4))
    varied = [
        builder.scattered(a_position(), values, 1, random.Random(seed), scalars={"turn": ["white", "black"]})
        for seed in range(8)
    ]

    assert kept.model("turn").value == "white", "left as it was, which is the trap"
    assert {one.model("turn").value for one in varied} == {"white", "black"}


def test_a_grid_nobody_named_a_value_for_is_left_empty():
    """Scattering fills what it was told about. A grid it was told nothing about has nothing to put there."""
    scattered = PositionBuilder().scattered(
        a_position(), {"piece": ["rook"]}, pieces=2, rng=random.Random(5)
    )

    assert all(scattered.model("colour").at(one) is None for one in scattered.model("colour").coordinates())


def test_a_position_with_no_grid_the_values_name_comes_back_emptied():
    scattered = PositionBuilder().scattered(
        a_position(), {"lamp": ["on"]}, pieces=2, rng=random.Random(6)
    )

    assert all(scattered.model("piece").at(one) is None for one in scattered.model("piece").coordinates())
