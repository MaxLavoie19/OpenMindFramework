from collections.abc import Iterator
from typing import Protocol

from openmind.doxastic.model.record import Record


class RecordStore(Protocol):
    """Where records are kept word for word, whichever way a project keeps them: a file of JSON lines, an SQLite
    database, or anything else behind this port.

    A store is append-only as far as the knowledge base is concerned: remembering the same id again writes the record
    anew, and `load` gives the last written of each id, in the order they were first written. `place` is whatever the
    store needs to find one record again — a byte offset in a file, a row id in a database — and the knowledge base
    keeps it without looking inside it."""

    def append(self, record: Record) -> object:
        """Writes the record and gives back its place, which `read` takes to find it again."""
        ...

    def read(self, place: object) -> Record:
        """The record at that place, word for word."""
        ...

    def load(self) -> Iterator[tuple[object, Record]]:
        """Every record kept, with its place, in the order they were remembered; a forgotten record is left out."""
        ...

    def forget(self, record_id: str) -> None:
        """Marks the record forgotten: `load` leaves it out from then on."""
        ...
