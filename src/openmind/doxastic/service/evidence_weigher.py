from collections.abc import Iterable

from openmind.doxastic.model.belief import Belief
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.record import Record


class EvidenceWeigher:
    """Which records bear on a claim, and which way.

    It decides what counts; the belief it gives does the arithmetic. A record bears on the claim when it is that claim's
    — the same name, the same holder, the same subjects — and says which way it goes; a record that only remembers
    something, with nothing for or against, is left out of both sides. What the records carry stays as they were
    written: nothing is discounted here yet, so weighing a teller by how reliable they have been belongs to this class
    when it comes."""

    def weigh(self, claim: Claim, records: Iterable[Record]) -> Belief:
        """Where the claim stands, given those records."""
        bearing = [record for record in records if record.bears and record.claim is not None and record.claim.key == claim.key]
        supporting = tuple(record for record in bearing if record.supports)
        opposing = tuple(record for record in bearing if not record.supports)
        return Belief(claim, supporting, opposing)
