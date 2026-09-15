from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DroppedCall:
    """What a droppable call gives instead of its result when it took a worker over its memory cap in a fresh worker too:
    the call's index in the argument lists, and the diagnosis of the second worker, None when that worker was ended from
    outside."""

    index: int
    diagnosis: Path | None
