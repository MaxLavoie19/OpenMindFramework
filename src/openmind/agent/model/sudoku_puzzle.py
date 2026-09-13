from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SudokuPuzzle:
    """A published puzzle: its collection, its position in the collection counting from 1, and its grid, 81 characters
    row by row with "." for an empty cell."""

    collection: str
    number: int
    grid: str
