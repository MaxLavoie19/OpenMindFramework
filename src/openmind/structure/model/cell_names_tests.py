import pytest

from openmind.structure.model.cell_names import CellNames
from openmind.structure.model.grid import Grid

#: Chess's names: the files a to h across, the ranks 8 down to 1, so row 1 is rank 8.
CHESS = CellNames(tuple("abcdefgh"), tuple("87654321"))

EMPTY = None


def test_a_cell_s_name_is_its_column_s_and_its_row_s() -> None:
    assert CHESS.to_coordinates("e4") == (5, 5)
    assert CHESS.to_alias((5, 5)) == "e4"
    assert (CHESS.to_coordinates("a8"), CHESS.to_coordinates("h1")) == ((1, 1), (8, 8))


def test_names_of_more_than_one_character_are_read_whole() -> None:
    spreadsheet = CellNames(("A", "B", "AA"), ("1", "2", "10"))

    assert spreadsheet.to_coordinates("AA10") == (3, 3)
    assert spreadsheet.to_alias((3, 3)) == "AA10"


def test_a_name_no_cell_has_is_refused() -> None:
    with pytest.raises(ValueError, match="No cell is named 'j9'"):
        CHESS.to_coordinates("j9")
    with pytest.raises(KeyError, match="outside"):
        CHESS.to_alias((9, 1))


def test_a_grid_named_this_way_is_read_by_name_and_reads_back_from_what_it_prints() -> None:
    board = Grid.filled((8, 8), EMPTY, CHESS).placed("e4", "white pawn")

    assert board[5, 5] == "white pawn" and board.at("e4") == "white pawn"
    assert eval(repr(board)) == board  # noqa: S307 - what a game declares is read back this way
