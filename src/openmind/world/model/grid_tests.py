import pickle

from openmind.world.model.grid import Grid

#: A 3 by 3 board: X on (1, 1) and (3, 3), O on (2, 2), the rest empty.
BOARD = Grid({(rank, file): None for rank in range(1, 4) for file in range(1, 4)})
BOARD.update({(1, 1): "X", (2, 2): "O", (3, 3): "X"})
#: A row of 5 cells keyed by whole numbers.
ROW = Grid({1: "A", 2: None, 3: "B", 4: None, 5: "A"})


def test_where_gives_the_coordinates_holding_a_value_in_order() -> None:
    assert (BOARD.where("X"), BOARD.where("O"), ROW.where("A"), BOARD.where("Z")) == (((1, 1), (3, 3)), ((2, 2),), (1, 5), ())


def test_parity_distance_and_steps_read_coordinates() -> None:
    assert (BOARD.parity((1, 1)), BOARD.parity((1, 2)), ROW.parity(3)) == (0, 1, 1)
    assert (BOARD.distance((1, 1), (3, 2)), BOARD.steps((1, 1), (3, 2)), ROW.distance(1, 4)) == (2, 3, 3)


def test_cells_on_a_row_a_column_or_a_diagonal_are_aligned_and_have_values_between() -> None:
    assert [BOARD.aligned((1, 1), other) for other in ((1, 3), (3, 1), (3, 3), (3, 2), (1, 1))] == [True, True, True, False, False]
    assert (BOARD.between((1, 1), (3, 3)), BOARD.between((3, 3), (1, 3)), BOARD.between((1, 1), (3, 2))) == (("O",), (None,), ())
    assert (ROW.between(1, 5), ROW.between(2, 3)) == ((None, "B", None), ())


def test_a_ray_steps_until_it_leaves_the_grid_and_neighbours_surround_a_cell() -> None:
    assert (BOARD.ray((1, 1), (1, 1)), BOARD.ray((1, 1), (-1, 0)), BOARD.ray((1, 1), (0, 0))) == (("O", "X"), (), ())
    assert (ROW.ray(3, -1), ROW.ray(3, (1,))) == ((None, "A"), (None, "A"))
    assert (sorted(map(str, BOARD.neighbours((1, 1)))), len(BOARD.neighbours((2, 2))), ROW.neighbours(1)) == (
        ["None", "None", "O"],
        8,
        (None,),
    )


def test_a_grid_is_a_dict_and_goes_to_another_process() -> None:
    copy = pickle.loads(pickle.dumps(BOARD))

    assert copy == dict(BOARD) and isinstance(copy, Grid) and copy.where("O") == ((2, 2),)
