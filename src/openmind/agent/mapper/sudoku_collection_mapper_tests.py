import pytest

from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.model.sudoku_puzzle import SudokuPuzzle

FIRST = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"
SECOND = "4.....8.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......"


def test_one_puzzle_per_line_is_numbered_from_one_and_blank_lines_are_skipped() -> None:
    assert SudokuCollectionMapper().to_puzzles("top95", f"{FIRST}\n\n{SECOND}\n") == (
        SudokuPuzzle("top95", 1, FIRST),
        SudokuPuzzle("top95", 2, SECOND),
    )


def test_a_grid_header_and_nine_rows_are_one_puzzle_with_zeros_as_empty_cells() -> None:
    rows = "\n".join(FIRST[index : index + 9].replace(".", "0") for index in range(0, 81, 9))

    assert SudokuCollectionMapper().to_puzzles("euler", f"Grid 01\n{rows}\nGrid 02\n{rows}\n") == (
        SudokuPuzzle("euler", 1, FIRST),
        SudokuPuzzle("euler", 2, FIRST),
    )


def test_a_line_of_the_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="top95, puzzle 2: needs 81 characters"):
        SudokuCollectionMapper().to_puzzles("top95", f"{FIRST}\n{SECOND[:80]}\n")


def test_a_character_that_is_neither_a_digit_nor_an_empty_mark_raises() -> None:
    with pytest.raises(ValueError, match="top95, puzzle 1: needs 81 characters"):
        SudokuCollectionMapper().to_puzzles("top95", FIRST.replace("5", "x", 1))


def test_a_grid_header_without_nine_full_rows_raises() -> None:
    with pytest.raises(ValueError, match="euler, puzzle 1: 'Grid 01' needs 9 rows of 9 characters"):
        SudokuCollectionMapper().to_puzzles("euler", "Grid 01\n530070000\n")
