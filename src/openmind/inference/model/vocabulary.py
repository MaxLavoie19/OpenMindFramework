from dataclasses import dataclass, field

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Vocabulary:
    """What the rows' positions hold, read once to generate expressions from: the players; every variable with the
    values seen, by model name and index — a scalar's index empty, a grid cell's its coordinates, a map entry's its key
    alone; every grid and map with the values seen and the indices it has; the names of the grids among them; and, per
    number of dimensions, every offset between two cells of one grid."""

    players: tuple[str, ...]
    values_by_variable: dict[tuple[str, tuple[Value, ...]], tuple[Value, ...]]
    values_by_base: dict[str, tuple[Value, ...]]
    indices_by_base: dict[str, frozenset[tuple[object, ...]]]
    offsets_by_arity: dict[int, tuple[tuple[int, ...], ...]]
    grids: frozenset[str] = frozenset()
    #: What the actions are called, and for each the values every parameter of it was seen to take.
    #:
    #: **Only a move rule reads these.** A position is worth what it is worth whoever is about to move, so a
    #: position term has no action to read and this is empty for one. A move term reads the action as well as
    #: the board, which is what lets it rate thirty-five moves in one reading where valuing what each leads to
    #: costs thirty-five.
    #:
    #: Read the way a move rule is already read at play time — `RuleHeuristic.rate` binds `action` to the
    #: action's name and every parameter under its own name — so a term fitted here is a term that can be run
    #: there, which is not true of anything this had to invent a convention for.
    values_by_parameter: dict[str, tuple[Value, ...]] = field(default_factory=dict)
    #: Every name an action was seen under.
    action_names: tuple[str, ...] = ()
