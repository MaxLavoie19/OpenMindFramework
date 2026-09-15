from dataclasses import dataclass

from openmind.world.model.value import Value


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """What the rows' positions hold, read once to generate expressions from: the players and the variable naming the
    player to act; every variable with the values seen; every indexed base with the values seen and the indices it has;
    and, per number of whole-number indices, every offset between two indices of one base."""

    players: tuple[str, ...]
    to_act: str
    values_by_variable: dict[str, tuple[Value, ...]]
    values_by_base: dict[str, tuple[Value, ...]]
    indices_by_base: dict[str, frozenset[tuple[object, ...]]]
    offsets_by_arity: dict[int, tuple[tuple[int, ...], ...]]
