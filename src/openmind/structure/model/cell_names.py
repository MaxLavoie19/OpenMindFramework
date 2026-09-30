from dataclasses import dataclass
from functools import lru_cache

from openmind.structure.model.coordinates import Coordinates


@lru_cache(maxsize=None)
def _coordinates(columns: tuple[str, ...], rows: tuple[str, ...]) -> dict[str, Coordinates]:
    """Every cell name of that naming, to the cell it points at.

    Kept beside the class rather than in it so that `CellNames` goes on holding nothing but the names —
    hashable, comparable, and picklable to a worker exactly as before. One naming is shared by every state of
    a game, so the cache holds one entry per game rather than one per position.

    **A name is a column's name then a row's, and nothing says a column's name cannot be a prefix of
    another's.** Built the same way `to_alias` writes them, so the two cannot disagree about what a cell is
    called; where two names would collide the later column wins, which is the same answer the scan gave.
    """
    return {
        f"{column_named}{row_named}": (row, column)
        for column, column_named in enumerate(columns, start=1)
        for row, row_named in enumerate(rows, start=1)
    }


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
        """The cell those names point at. A name no cell has raises ValueError.

        **Looked up rather than searched, because this is the hottest line in the run.** Written as a scan it
        tried every column with `startswith` and then every row — up to sixty-four string comparisons a call.
        Profiled over one judging it ran 1.4 million times and did 6.3 million `startswith`, about a third of
        the whole cost, and the judging is what blocks the learner.

        The names are the only thing a cell's coordinates depend on, so the whole mapping is built once per
        set of names and shared by every state that carries them. `CellNames` still holds nothing but the
        names, so it stays hashable and still travels to a worker process."""
        found = _coordinates(self.columns, self.rows).get(alias)
        if found is None:
            raise ValueError(f"No cell is named {alias!r}")
        return found

    def to_alias(self, where: Coordinates) -> str:
        """What the game calls that cell: its column's name and its row's. A cell outside them raises KeyError."""
        row, column = where
        if not 1 <= row <= len(self.rows) or not 1 <= column <= len(self.columns):
            raise KeyError(f"{where} is outside the cells named by {len(self.columns)} columns and {len(self.rows)} rows")
        return f"{self.columns[column - 1]}{self.rows[row - 1]}"
