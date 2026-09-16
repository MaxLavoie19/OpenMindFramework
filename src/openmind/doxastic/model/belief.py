from dataclasses import dataclass
from functools import reduce

from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.record import Record


def accumulated(records: tuple[Record, ...]) -> float:
    """How much a pile of records carries together, from 0 to 1: each one leaves what the others haven't covered, so
    `1 − Π(1 − strength)`. One conclusive record is enough; many weak ones add up without ever reaching certainty."""
    return 1.0 - reduce(lambda left, record: left * (1.0 - min(max(record.strength, 0.0), 1.0)), records, 1.0)


@dataclass(frozen=True, slots=True)
class Belief:
    """Where a claim stands: the records for it and the records against it, kept apart.

    The two sides are never folded into one number, so poor evidence for a claim and solid evidence against it read as
    exactly that. `belief` is what the supporting records carry, `disbelief` what the opposing ones carry, `uncertainty`
    what neither side covers, and `contested` how much both sides carry at once."""

    claim: Claim
    supporting: tuple[Record, ...] = ()
    opposing: tuple[Record, ...] = ()

    @property
    def belief(self) -> float:
        return accumulated(self.supporting)

    @property
    def disbelief(self) -> float:
        return accumulated(self.opposing)

    @property
    def uncertainty(self) -> float:
        """What neither side covers; 1 with no record at all, 0 once the two sides cover everything between them."""
        return max(0.0, 1.0 - self.belief - self.disbelief)

    @property
    def contested(self) -> float:
        """How much the claim is held both ways at once: the smaller of belief and disbelief."""
        return min(self.belief, self.disbelief)

    @property
    def net(self) -> float:
        """Belief less disbelief, from -1 to 1."""
        return self.belief - self.disbelief

    @property
    def by_source(self) -> tuple[tuple[str, float, float], ...]:
        """What each kind of source carries, as (source, belief, disbelief), by source name: so that "proved against,
        merely told for" reads as that, and not as one number."""
        sources = sorted({record.provenance.source for record in (*self.supporting, *self.opposing)})
        return tuple(
            (
                source,
                accumulated(tuple(record for record in self.supporting if record.provenance.source == source)),
                accumulated(tuple(record for record in self.opposing if record.provenance.source == source)),
            )
            for source in sources
        )
