from openmind.structure.model.grid import Grid
from openmind.structure.model.scalar import Scalar
from openmind.world.constant.grid_text_constant import EMPTY_MARK
from openmind.world.model.state import State


class GridTextMapper:
    """Maps a state to readable text. Every two-dimensional grid is laid out: its name and the column numbers on top,
    then one line per row starting with its number, with "." for None. Every other model follows as a "name = value"
    line."""

    def to_text(self, state: State) -> str:
        grids: list[str] = []
        lines: list[str] = []
        for name, model in state.models:
            if isinstance(model, Grid) and len(model.shape) == 2:
                grids.append(self._grid(name, model))
            elif isinstance(model, Scalar):
                lines.append(f"{name} = {model.value!r}")
            else:
                lines.append(f"{name} = {model!r}")
        return "\n".join([*grids, *lines])

    def _grid(self, name: str, grid: Grid) -> str:
        rows, cols = range(1, grid.shape[0] + 1), range(1, grid.shape[1] + 1)
        texts = {where: EMPTY_MARK if value is None else str(value) for where, value in grid.items()}
        width = max(len(text) for text in [*texts.values(), *(str(col) for col in cols)])
        label = max(len(text) for text in [name, *(str(row) for row in rows)])
        header = " ".join([name.rjust(label), *(str(col).rjust(width) for col in cols)])
        body = [" ".join([str(row).rjust(label), *(texts[(row, col)].rjust(width) for col in cols)]) for row in rows]
        return "\n".join([header, *body])
