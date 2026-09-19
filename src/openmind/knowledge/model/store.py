from collections.abc import Iterator
from typing import Protocol


class Store(Protocol):
    """Where one kind of knowledge is kept, whichever way an integrator keeps it: JSON lines by default, their own
    long-term storage otherwise. Entries come as JSON-ready dicts, each with an `id`.

    The store is append-only as far as the knowledge base is concerned: writing an id again writes the entry anew, and
    `load` gives the last written of each id, in the order the ids were first written. The integrator owns the data's
    lifecycle."""

    def append(self, entry: dict[str, object]) -> None:
        """Writes the entry."""
        ...

    def load(self) -> Iterator[dict[str, object]]:
        """Every entry kept, the last written of each id, in the order the ids were first written; a forgotten one is
        left out."""
        ...

    def forget(self, entry_id: str) -> None:
        """Marks the entry forgotten: `load` leaves it out from then on."""
        ...
