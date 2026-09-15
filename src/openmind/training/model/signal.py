from dataclasses import dataclass

from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class Signal:
    """Something that reads a position for a player, by name: winning itself, without a source or parts; an expression's
    value for the player, from its source; or an aggregation voting with the signals its parts name. A signal deduced
    from the rules names its premises, the signals it follows from."""

    name: str
    source: PythonRule | None = None
    parts: tuple[str, ...] = ()
    premises: tuple[str, ...] = ()
