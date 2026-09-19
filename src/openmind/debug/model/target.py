from dataclasses import dataclass

from openmind.structure.model.value import Value


@dataclass(frozen=True, slots=True)
class Target:
    """What a verbosity setting applies to: log lines from a package or service (a logger name or its prefix), or those
    written while a reasoning frame of a context, of a kind, or carrying tags is open. Every field given must match; None
    and no tags match anything."""

    logger: str | None = None
    context: str | None = None
    frame: str | None = None
    tags: tuple[tuple[str, Value], ...] = ()
