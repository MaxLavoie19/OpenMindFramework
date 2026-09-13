from pathlib import Path

from openmind.agent.constant.sudoku_constant import COLLECTION_SUFFIX
from openmind.agent.mapper.sudoku_collection_mapper import SudokuCollectionMapper
from openmind.agent.model.sudoku_puzzle import SudokuPuzzle


class SudokuPuzzleRepository:
    """Loads published sudoku collections, one text file per collection, named after the collection."""

    def __init__(self, sudoku_collection_mapper: SudokuCollectionMapper) -> None:
        self._sudoku_collection_mapper = sudoku_collection_mapper

    def collections(self, directory: Path) -> tuple[str, ...]:
        """The names of the directory's <collection>.txt files, sorted; none when the directory is missing."""
        return tuple(sorted(path.stem for path in directory.glob(f"*{COLLECTION_SUFFIX}")))

    def load(self, directory: Path, collection: str) -> tuple[SudokuPuzzle, ...]:
        path = directory / f"{collection}{COLLECTION_SUFFIX}"
        return self._sudoku_collection_mapper.to_puzzles(collection, path.read_text(encoding="utf-8"))
