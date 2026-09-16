from dataclasses import dataclass

from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.provenance import Provenance


@dataclass(frozen=True, slots=True)
class Record:
    """One thing the agent remembers, kept word for word.

    `text` is what was said, seen or found, as it was; nothing rewrites it. `subjects` are who or what it is about,
    `names` the names it mentions and `keywords` the words it can be looked up by; all three are what the knowledge base
    indexes. Where the record bears on a claim, `supports` says which way — True for the claim, False for its negation,
    None where it only records something — and `strength` from 0 to 1 how much it carries, 1 being conclusive. `count`
    is how many cases a counted record stands for; every other kind stands for itself, one.

    `id` is empty until the knowledge base remembers the record, which gives it one; from then on the record can be
    cited by it and read back word for word."""

    text: str
    provenance: Provenance
    subjects: tuple[str, ...] = ()
    names: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    claim: Claim | None = None
    supports: bool | None = None
    strength: float = 1.0
    count: int = 1
    id: str = ""

    @property
    def bears(self) -> bool:
        """Whether the record says anything for or against its claim."""
        return self.claim is not None and self.supports is not None
