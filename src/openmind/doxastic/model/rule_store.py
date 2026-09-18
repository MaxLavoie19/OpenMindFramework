from collections.abc import Iterator
from typing import Protocol

from openmind.doxastic.model.rule_record import RuleRecord


class RuleStore(Protocol):
    """Where the rules the agent knows are kept, whichever way a project keeps them: a file of JSON lines, an SQLite
    database, or anything else behind this port.

    A store is append-only as far as the knowledge base is concerned: declaring the same id again writes the rule anew,
    and `load` gives the last written of each id, in the order they were first written. `place` is whatever the store
    needs to find one rule again, which the knowledge base keeps without looking inside it."""

    def append(self, rule: RuleRecord) -> object:
        """Writes the rule and gives back its place, which `read` takes to find it again."""
        ...

    def read(self, place: object) -> RuleRecord:
        """The rule at that place."""
        ...

    def load(self) -> Iterator[tuple[object, RuleRecord]]:
        """Every rule kept, with its place, in the order they were declared; a forgotten rule is left out."""
        ...

    def forget(self, rule_id: str) -> None:
        """Marks the rule forgotten: `load` leaves it out from then on."""
        ...
