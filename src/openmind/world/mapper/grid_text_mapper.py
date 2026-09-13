from openmind.world.constant.grid_text_constant import EMPTY_MARK, MISSING_MARK
from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.state import State
from openmind.world.model.value import Value


class GridTextMapper:
    """Maps a state to readable text. Variables named <base>(<row>,<col>) with whole-number indices become one grid per
    base: the base and the column numbers on top, then one line per row starting with its number, with "." for None and
    a blank for a missing cell. Every other variable follows as a "name = value" line."""

    def __init__(self, variable_name_mapper: VariableNameMapper) -> None:
        self._variable_name_mapper = variable_name_mapper

    def to_text(self, state: State) -> str:
        grids: dict[str, dict[tuple[int, int], Value]] = {}
        lines: list[str] = []
        for name, value in state.variables:
            base, indices = self._variable_name_mapper.from_name(name)
            if len(indices) == 2 and all(index.isdecimal() for index in indices):
                grids.setdefault(base, {})[(int(indices[0]), int(indices[1]))] = value
            else:
                lines.append(f"{name} = {value!r}")
        return "\n".join([*(self._grid(base, cells) for base, cells in grids.items()), *lines])

    def _grid(self, base: str, cells: dict[tuple[int, int], Value]) -> str:
        rows = range(min(row for row, _ in cells), max(row for row, _ in cells) + 1)
        cols = range(min(col for _, col in cells), max(col for _, col in cells) + 1)
        texts = {position: EMPTY_MARK if value is None else str(value) for position, value in cells.items()}
        width = max(len(text) for text in [*texts.values(), *(str(col) for col in cols)])
        label = max(len(text) for text in [base, *(str(row) for row in rows)])
        header = " ".join([base.rjust(label), *(str(col).rjust(width) for col in cols)])
        body = [
            " ".join([str(row).rjust(label), *(texts.get((row, col), MISSING_MARK).rjust(width) for col in cols)])
            for row in rows
        ]
        return "\n".join([header, *body])
