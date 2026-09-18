import pickle

from openmind.world.model.grid import Grid

#: A 3 by 3 board: X on (1, 1) and (3, 3), O on (2, 2), the rest empty.
BOARD = Grid({(rank, file): None for rank in range(1, 4) for file in range(1, 4)})
BOARD.update({(1, 1): "X", (2, 2): "O", (3, 3): "X"})
#: A row of 5 cells keyed by whole numbers.
ROW = Grid({1: "A", 2: None, 3: "B", 4: None, 5: "A"})


def test_where_gives_the_coordinates_holding_a_value_in_order() -> None:
    assert (BOARD.where("X"), BOARD.where("O"), ROW.where("A"), BOARD.where("Z")) == (((1, 1), (3, 3)), ((2, 2),), (1, 5), ())


def test_a_grid_is_a_dict_and_goes_to_another_process() -> None:
    copy = pickle.loads(pickle.dumps(BOARD))

    assert copy == dict(BOARD) and isinstance(copy, Grid) and copy.where("O") == ((2, 2),)
