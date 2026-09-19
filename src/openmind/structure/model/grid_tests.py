import pytest

from openmind.structure.constant.direction_constant import DIAGONAL, MANHATTAN, ORTHOGONAL
from openmind.structure.model.coordinates import Coordinates
from openmind.structure.model.grid import Grid

EMPTY = None


class ChessSquares:
    """Chess's names for an 8 by 8 grid, rank 8 at the top row."""

    def to_coordinates(self, alias: str) -> Coordinates:
        return (9 - int(alias[1]), "abcdefgh".index(alias[0]) + 1)

    def to_alias(self, where: Coordinates) -> str:
        return f"{'abcdefgh'[where[1] - 1]}{9 - where[0]}"

    def __hash__(self) -> int:
        return hash(ChessSquares)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ChessSquares)


def test_a_grid_is_read_and_changed_by_coordinates_from_1_row_first_and_every_change_is_a_new_grid() -> None:
    grid = Grid.filled((3, 3), EMPTY)

    changed = grid.placed((2, 3), "X")

    assert (changed[2, 3], changed.at((1, 1)), grid[2, 3]) == ("X", EMPTY, EMPTY)
    assert changed != grid and hash(changed) == hash(Grid.filled((3, 3), EMPTY).placed((2, 3), "X"))
    assert changed.where("X") == ((2, 3),)


def test_a_grid_of_the_wrong_number_of_cells_or_read_outside_is_refused() -> None:
    with pytest.raises(ValueError, match="9 cells"):
        Grid((3, 3), (EMPTY,) * 8)
    with pytest.raises(KeyError, match="outside"):
        Grid.filled((3, 3), EMPTY)[4, 1]


def test_a_game_s_aliases_name_cells_as_it_does() -> None:
    board = Grid.filled((8, 8), EMPTY, ChessSquares()).placed("e4", "white pawn")

    assert board[5, 5] == "white pawn" and board.at("e4") == "white pawn"
    assert board.alias((8, 1)) == "a1"
    assert board.moved("e4", "e5", EMPTY).where("white pawn") == ((4, 5),)


def test_lines_follow_the_grid_s_directions_each_counted_once() -> None:
    full = Grid.filled((3, 3), EMPTY)
    straight = Grid.filled((3, 3), EMPTY, directions=ORTHOGONAL)

    assert len(full.lines(3)) == 8
    assert len(straight.lines(3)) == 6
    assert len(Grid.filled((6, 7), EMPTY).lines(4)) == 69
    assert full.lines_through((2, 2), 3) == tuple(line for line in full.lines(3) if (2, 2) in line)
    assert len(full.lines_through((2, 2), 3)) == 4


def test_neighbours_rows_columns_diagonals_and_boxes() -> None:
    grid = Grid.filled((3, 3), EMPTY)

    assert set(grid.neighbours((1, 1))) == {(1, 2), (2, 1), (2, 2)}
    assert set(grid.neighbours((1, 1), diagonal_too=False)) == {(1, 2), (2, 1)}
    assert grid.rows()[0] == ((1, 1), (1, 2), (1, 3)) and grid.columns()[2] == ((1, 3), (2, 3), (3, 3))
    assert grid.diagonals() == (((1, 1), (2, 2), (3, 3)), ((1, 3), (2, 2), (3, 1)))
    assert len(Grid.filled((9, 9), EMPTY).boxes((3, 3))) == 9
    assert Grid.filled((9, 9), EMPTY).boxes((3, 3))[1][0] == (1, 4)


def test_a_ray_runs_to_the_first_blocked_cell_or_the_edge() -> None:
    board = Grid.filled((8, 8), EMPTY).placed((5, 1), "black knight")

    assert board.ray((8, 1), (-1, 0), lambda held: held is not EMPTY) == ((7, 1), (6, 1), (5, 1))
    assert board.ray((8, 8), (-1, 0), lambda held: held is not EMPTY)[-1] == (1, 8)


def test_distance_is_a_king_s_or_a_rook_s_steps() -> None:
    grid = Grid.filled((8, 8), EMPTY)

    assert grid.distance((1, 1), (3, 4)) == 3
    assert grid.distance((1, 1), (3, 4), MANHATTAN) == 5


def test_a_grid_turns_and_mirrors() -> None:
    grid = Grid((2, 3), ("a", "b", "c", "d", "e", "f"))

    assert grid.rotated() == Grid((3, 2), ("d", "a", "e", "b", "f", "c"))
    assert grid.rotated(4) == grid
    assert grid.reflected(1) == Grid((2, 3), ("c", "b", "a", "f", "e", "d"))
    assert grid.reflected(0) == Grid((2, 3), ("d", "e", "f", "a", "b", "c"))


def test_a_grid_has_any_number_of_dimensions() -> None:
    cube = Grid.filled((3, 3, 3), EMPTY)

    assert len(cube.coordinates()) == 27
    assert len(cube.neighbours((2, 2, 2))) == 26
    assert len(cube.lines(3)) == 49
    assert len(DIAGONAL) == 4 and len(ORTHOGONAL) == 4
