from openmind.agent.constant.sudoku_constant import CELLS, CLUE_MARKS, EMPTY_MARK, EULER_EMPTY_MARK, EULER_HEADER, SIZE
from openmind.agent.model.sudoku_puzzle import SudokuPuzzle


class SudokuCollectionMapper:
    """Reads a published collection of sudoku puzzles, numbered from 1 in file order. It takes two formats: one
    81-character puzzle per line (Norvig's collections), or a "Grid NN" line followed by 9 rows of 9 characters (Project
    Euler's), with "." or "0" marking an empty cell. Blank lines are skipped."""

    def to_puzzles(self, collection: str, text: str) -> tuple[SudokuPuzzle, ...]:
        """The collection's puzzles, with "." for every empty cell; a puzzle that doesn't fit either format raises
        ValueError."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        marks = (EMPTY_MARK, EULER_EMPTY_MARK, *CLUE_MARKS)
        puzzles: list[SudokuPuzzle] = []
        index = 0
        while index < len(lines):
            number = len(puzzles) + 1
            if lines[index].startswith(EULER_HEADER):
                rows = lines[index + 1 : index + 1 + SIZE]
                if len(rows) < SIZE or any(len(row) != SIZE for row in rows):
                    raise ValueError(
                        f"{collection}, puzzle {number}: {lines[index]!r} needs {SIZE} rows of {SIZE} characters"
                    )
                grid = "".join(rows)
                index += 1 + SIZE
            else:
                grid = lines[index]
                index += 1
            if len(grid) != CELLS or any(mark not in marks for mark in grid):
                raise ValueError(
                    f"{collection}, puzzle {number}: needs {CELLS} characters, each {EMPTY_MARK!r}, "
                    f"{EULER_EMPTY_MARK!r} or a digit from 1 to {SIZE}: {grid!r}"
                )
            puzzles.append(SudokuPuzzle(collection, number, grid.replace(EULER_EMPTY_MARK, EMPTY_MARK)))
        return tuple(puzzles)
