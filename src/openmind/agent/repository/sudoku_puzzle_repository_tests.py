from pathlib import Path

from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.model.sudoku_puzzle import SudokuPuzzle
from openmind.agent.repository.sudoku_puzzle_repository import SudokuPuzzleRepository

FIRST = "53..7....6..195....98....6.8...6...34..8.3..17...2...6.6....28....419..5....8..79"


def test_collections_are_the_text_files_sorted_by_name(tmp_path: Path) -> None:
    for name in ("top95.txt", "euler.txt", "notes.md"):
        (tmp_path / name).write_text(f"{FIRST}\n", encoding="utf-8")

    assert SudokuPuzzleRepository(SudokuCollectionMapper()).collections(tmp_path) == ("euler", "top95")


def test_a_missing_directory_has_no_collection(tmp_path: Path) -> None:
    assert SudokuPuzzleRepository(SudokuCollectionMapper()).collections(tmp_path / "missing") == ()


def test_load_reads_the_puzzles_of_a_collection(tmp_path: Path) -> None:
    (tmp_path / "top95.txt").write_text(f"{FIRST}\n", encoding="utf-8")

    assert SudokuPuzzleRepository(SudokuCollectionMapper()).load(tmp_path, "top95") == (
        SudokuPuzzle("top95", 1, FIRST),
    )
