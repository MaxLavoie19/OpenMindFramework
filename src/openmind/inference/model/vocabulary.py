from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """What the rows' positions hold, read once to generate expressions from: the players and the model saying who acts;
    every variable with the values seen, by model name and index — a scalar's index empty, a grid cell's its
    coordinates, a map entry's its key alone; every grid and map with the values seen and the indices it has; the names
    of the grids among them; and, per number of dimensions, every offset between two cells of one grid."""

    players: tuple[str, ...]
    to_act: str
    values_by_variable: dict[tuple[str, tuple[Value, ...]], tuple[Value, ...]]
    values_by_base: dict[str, tuple[Value, ...]]
    indices_by_base: dict[str, frozenset[tuple[object, ...]]]
    offsets_by_arity: dict[int, tuple[tuple[int, ...], ...]]
    grids: frozenset[str] = frozenset()
