import logging
import re
from collections.abc import Mapping, Sequence

from openmind.inference.model.example import Example
from openmind.inference.service.action_readings import (
    AFTER,
    AT,
    BELONGS,
    BETWEEN,
    COLUMN,
    COLUMNS,
    COLUMNS_APART,
    DIAGONAL,
    DISTANCE,
    FORWARD,
    REACHED,
    REACHED_HOLDING,
    REACHED_OWNED,
    REACHES,
    ROW,
    ROW_FROM_SIDE,
    ROWS,
    ROWS_APART,
    SAME,
    STRAIGHT,
)
from openmind.rule.model.literal import Literal
from openmind.rule.model.term import Constant
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: Each reading's template with the name it is known by once its slots are opened up.
#:
#: The name is what the reading *is*, with the things it is said of taken out of it: `"{model} at {parameter}"`
#: read of a piece at a source is `at(piece, source, ...)`. Every one of these is generic — a thing, a place, a
#: player, a side — and a game that has no grid simply never produces the ones about rows and columns.
TEMPLATES: tuple[tuple[str, str], ...] = (
    (BELONGS, "belongs"),
    (REACHED_OWNED, "can reach owned"),
    (REACHED, "can reach holding"),
    (REACHED_HOLDING, "can reach holding"),
    (REACHES, "can reach"),
    (ROW_FROM_SIDE, "row from own side"),
    (FORWARD, "rows forward"),
    (ROWS_APART, "rows apart"),
    (COLUMNS_APART, "columns apart"),
    (ROWS, "rows"),
    (COLUMNS, "columns"),
    (STRAIGHT, "share a row or a column"),
    (DIAGONAL, "on a diagonal"),
    (DISTANCE, "steps"),
    (BETWEEN, "things between"),
    (SAME, "is"),
    (ROW, "row"),
    (COLUMN, "column"),
    (AT, "at"),
)

#: A reading of the position the action leads to, which wraps another reading.
AFTERWARDS = "after it"


class ReadingLiterals:
    """The game's readings as literals, with their name templates opened up.

    This is the bridge, and it is where the whole rework pays off or does not.

    A reading is named by filling a template — `"{model} at {parameter}"` becomes `"piece at source"` — and the
    filled string is then used as a key. That throws away the very thing that made the reading general. Two
    readings that are the same question asked of different things look like two unrelated names, so a rule can
    only ever compare a name to a value, and a rule about a player's own side has to be written once per player
    and is then not about sides at all.

    Here the slots come back out as arguments. `"rows from source to target"` read as 3 becomes
    `rows(source, target, 3)`, and a clause may now put a variable where the 3 is, or where `source` is, or tie
    one reading's player to another's. That is one clause where there were several, and it is the reason for
    everything else in this package.

    Nothing here is any game's. The readings are OMF's own vocabulary — a thing, a place, a player, a side, what
    is between two places — and a game with no grid never produces the ones about rows, so never sees them.

    It keeps nothing: built once with the templates it knows, it is given readings on every call."""

    def __init__(self, templates: Sequence[tuple[str, str]] = TEMPLATES) -> None:
        self._templates = tuple((self._pattern(template), name, self._slots(template)) for template, name in templates)

    def literals(self, readings: Mapping[str, Value]) -> tuple[Literal, ...]:
        """Those readings as ground literals, each slot of each name become a term of its own.

        A reading whose name matches no template is kept whole, as a predicate of one term: an unknown reading is
        still evidence, and refusing it would lose what a game says in its own way."""
        found: list[Literal] = []
        for name, value in readings.items():
            found.append(self._literal(name, value))
        return tuple(found)

    def example(self, readings: Mapping[str, Value], holds: bool, where: object | None = None) -> Example:
        """One case to learn from: everything read of it, and whether the thing being learned held."""
        return Example(self.literals(readings), holds, where)

    def examples(
        self, seen: Sequence[tuple[Mapping[str, Value], bool]], within: Sequence[object] = ()
    ) -> tuple[Example, ...]:
        """Cases from readings already gathered, each with the position it was read in where there is one."""
        places = tuple(within) if len(within) == len(seen) else (None,) * len(seen)
        return tuple(self.example(readings, holds, place) for (readings, holds), place in zip(seen, places))

    def _literal(self, name: str, value: Value) -> Literal:
        said, afterwards = self._unwrapped(name)
        for pattern, predicate, slots in self._templates:
            found = pattern.fullmatch(said)
            if found is None:
                continue
            terms = [Constant(self._read(found.group(slot))) for slot in slots]
            terms.append(Constant(value))
            return Literal(f"{AFTERWARDS}, {predicate}" if afterwards else predicate, tuple(terms))
        return Literal(said if not afterwards else f"{AFTERWARDS}, {said}", (Constant(value),))

    def _unwrapped(self, name: str) -> tuple[str, bool]:
        """A reading of the position the action leads to, and the reading it wraps."""
        wrapper = AFTER.replace("{reading}", "")
        if name.startswith(wrapper):
            return name[len(wrapper) :], True
        return name, False

    def _pattern(self, template: str) -> re.Pattern[str]:
        """The template as something a filled name can be matched against, its slots becoming the parts to take."""
        pattern = ""
        for piece in re.split(r"(\{[a-z_]+(?:!r)?\})", template):
            if piece.startswith("{") and piece.endswith("}"):
                pattern += f"(?P<{piece.strip('{}').removesuffix('!r')}>.+?)"
            else:
                pattern += re.escape(piece)
        return re.compile(pattern)

    def _slots(self, template: str) -> tuple[str, ...]:
        return tuple(one.removesuffix("!r") for one in re.findall(r"\{([a-z_]+(?:!r)?)\}", template))

    def _read(self, said: str) -> Value:
        """What a slot held, as the value it was rather than as the text it was written as."""
        if said.startswith("'") and said.endswith("'") and len(said) > 1:
            return said[1:-1]
        if said == "None":
            return None
        try:
            return int(said)
        except ValueError:
            return said
