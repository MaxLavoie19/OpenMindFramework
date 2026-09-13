from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TicTacToeVariant:
    """A variant of tic-tac-toe: the grid's width and height, how many marks in a row win, and whether marks fall to the
    lowest empty cell of a column."""

    name: str
    width: int
    height: int
    line: int
    gravity: bool
