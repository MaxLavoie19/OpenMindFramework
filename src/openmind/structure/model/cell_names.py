from dataclasses import dataclass

from openmind.structure.model.coordinates import Coordinates


@dataclass(frozen=True, slots=True)
class CellNames:
    """How a game names a grid's cells: chess's a1 to h8, a spreadsheet's B3, Go's d4.

    A cell's name is its column's name followed by its row's name, so the names are given per column, from column 1,
    and per row, from row 1, the top. Chess names its columns a to h and its rows 8 down to 1, so row 1 column 5 is
    e8, and a board read row by row is a board as white sees it.

    It holds nothing but the names, so a state carrying it is hashable, travels to a worker process, and reads back
    from what a game declared: a grid named this way is an OMF data structure like any other."""

    columns: tuple[str, ...]
    rows: tuple[str, ...]

    def to_coordinates(self, alias: str) -> Coordinates:
        """The cell those names point at. A name no cell has raises ValueError."""
        for column, named in enumerate(self.columns, start=1):
            if not alias.startswith(named):
                continue
            rest = alias[len(named) :]
            for row, row_named in enumerate(self.rows, start=1):
                if rest == row_named:
                    return (row, column)
        raise ValueError(f"No cell is named {alias!r}")

    def to_alias(self, where: Coordinates) -> str:
        """What the game calls that cell: its column's name and its row's. A cell outside them raises KeyError."""
        row, column = where
        if not 1 <= row <= len(self.rows) or not 1 <= column <= len(self.columns):
            raise KeyError(f"{where} is outside the cells named by {len(self.columns)} columns and {len(self.rows)} rows")
        return f"{self.columns[column - 1]}{self.rows[row - 1]}"
