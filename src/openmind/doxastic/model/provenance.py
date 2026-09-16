from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a record came from: which kind of source (see `constant/doxastic_constant.py`), who said it where it was
    told, and the position, game, round and ply it arose in. `when` is the wall clock, set when the record is
    remembered."""

    source: str
    told: str | None = None
    at: str | None = None
    game: str | None = None
    round: int | None = None
    ply: int | None = None
    when: datetime | None = None
