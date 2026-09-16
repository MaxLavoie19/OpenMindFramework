import logging
from dataclasses import replace
from datetime import datetime

from openmind.doxastic.model.belief import Belief
from openmind.doxastic.model.claim import Claim
from openmind.doxastic.model.record import Record
from openmind.doxastic.model.record_store import RecordStore
from openmind.doxastic.service.evidence_weigher import EvidenceWeigher
from openmind.doxastic.service.recall_cache import RecallCache
from openmind.doxastic.service.record_index import RecordIndex

logger = logging.getLogger(__name__)

#: How many digits a record's id has before it needs more.
ID_DIGITS = 6


class KnowledgeBase:
    """What the agent remembers, and where each of its claims stands.

    Everything goes in as a record, word for word, with where it came from: what it was told and by whom, what it saw,
    what it did itself and what came of it, what it proved, what it counted over many games, and what holds only in a
    relaxed domain. A record can be looked up later by subject, by name mentioned, by keyword, by claim, by who told it,
    by kind of source, or cited by its id and read back exactly as it was written.

    Where records bear on a claim, the two sides are kept apart: the agent can hold poor evidence for a claim and solid
    evidence against it, and both show. Nothing is overwritten and nothing is averaged away.

    The store keeps the records; the index says where each one is, and holds ids rather than records; the cache holds
    the ones in context, so what the agent is working on is at hand and the rest stays on disk until asked for."""

    def __init__(
        self,
        domain: str,
        record_store: RecordStore,
        record_index: RecordIndex,
        recall_cache: RecallCache,
        evidence_weigher: EvidenceWeigher,
    ) -> None:
        self._domain = domain
        self._store = record_store
        self._index = record_index
        self._cache = recall_cache
        self._weigher = evidence_weigher
        self._last_id = 0
        self._load()

    @property
    def domain(self) -> str:
        return self._domain

    def __len__(self) -> int:
        return len(self._index)

    def remember(self, record: Record) -> Record:
        """Keeps the record, word for word, and gives it back with the id it can be cited by. A record already carrying
        an id is written anew under that id, which is how a counted record raises its count."""
        kept = record
        if not kept.id:
            self._last_id += 1
            kept = replace(kept, id=f"{self._last_id:0{ID_DIGITS}d}")
        if kept.provenance.when is None:
            kept = replace(kept, provenance=replace(kept.provenance, when=datetime.now()))
        place = self._store.append(kept)
        self._index.add(kept, place)
        self._cache.keep(kept)
        logger.debug(
            "Remembered %s %s: %s", kept.provenance.source, kept.id, kept.text[:200].replace("\n", " ")
        )
        return kept

    def cite(self, record_id: str) -> Record | None:
        """The record with that id, word for word, or None where nothing was remembered under it."""
        held = self._cache.get(record_id)
        if held is not None:
            return held
        place = self._index.place(record_id)
        if place is None:
            return None
        read = self._store.read(place)
        self._cache.keep(read)
        return read

    def recall(
        self,
        subject: str | None = None,
        name: str | None = None,
        keyword: str | None = None,
        claim: Claim | str | None = None,
        told: str | None = None,
        source: str | None = None,
        since: datetime | None = None,
        at: str | None = None,
        limit: int | None = None,
    ) -> tuple[Record, ...]:
        """The records answering every field named, oldest first; every record where no field is named. `since` keeps
        those remembered then or later, `at` those from one position, and `limit` the newest that many of them."""
        key = claim.key if isinstance(claim, Claim) else claim
        found = self._index.find(subject, name, keyword, key, told, source)
        records = [record for record in (self.cite(record_id) for record_id in found) if record is not None]
        if since is not None:
            records = [record for record in records if record.provenance.when is not None and record.provenance.when >= since]
        if at is not None:
            records = [record for record in records if record.provenance.at == at]
        return tuple(records[-limit:] if limit is not None else records)

    def attend(self, *words: str) -> int:
        """Brings everything under those subjects, names or keywords into context, so recalling them costs nothing
        while the agent works on them; gives how many records that is."""
        brought: set[str] = set()
        for word in words:
            for record_id in self._index.under(word):
                if self.cite(record_id) is not None:
                    brought.add(record_id)
        logger.info("Attending to %s: %d records, %d in context", ", ".join(words), len(brought), len(self._cache))
        return len(brought)

    def weigh(self, claim: Claim) -> Belief:
        """Where the claim stands: the records for it and the records against it, kept apart."""
        return self._weigher.weigh(claim, self.recall(claim=claim))

    def claims(self, subject: str | None = None, holder: tuple[str, ...] | None = None) -> tuple[Claim, ...]:
        """Every claim the records bear on, once each, in the order they were first claimed; those about a subject or
        held by that chain of players where either is named."""
        seen: dict[str, Claim] = {}
        for record in self.recall(subject=subject):
            if record.claim is not None and record.claim.key not in seen:
                if holder is None or record.claim.holder == holder:
                    seen[record.claim.key] = record.claim
        return tuple(seen.values())

    def contested(self, least: float = 0.2) -> tuple[Belief, ...]:
        """The claims held both ways at once, the most contested first: where the agent has evidence for a claim and
        evidence against it, both carrying at least that much."""
        beliefs = [self.weigh(claim) for claim in self.claims()]
        return tuple(sorted((belief for belief in beliefs if belief.contested >= least), key=lambda belief: -belief.contested))

    def forget(self, record_id: str) -> None:
        """Drops the record: it is left out of everything from then on."""
        record = self.cite(record_id)
        if record is None:
            return
        self._store.forget(record_id)
        self._index.remove(record)
        self._cache.drop(record_id)

    def _load(self) -> None:
        """Takes in what the store already holds: the index learns where every record is, and the ids carry on from the
        last one used."""
        for place, record in self._store.load():
            self._index.add(record, place)
            if record.id.isdigit():
                self._last_id = max(self._last_id, int(record.id))
        if len(self._index):
            logger.info("Knowledge of %s: %d records remembered", self._domain, len(self._index))
