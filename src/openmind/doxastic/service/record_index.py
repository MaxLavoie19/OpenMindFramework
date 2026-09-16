from collections import defaultdict

from openmind.doxastic.model.record import Record

#: The fields a record is indexed under, each holding words a lookup can name.
SUBJECT, NAME, KEYWORD, CLAIM, TOLD, SOURCE = "subject", "name", "keyword", "claim", "told", "source"


class RecordIndex:
    """Which records to find where: by subject, by name mentioned, by keyword, by claim, by who told it and by kind of
    source.

    Words are matched whatever their case — the record's own words stay as they were written; only the lookup is
    forgiving. A lookup naming several fields gives the records answering all of them. The index holds ids and places,
    never the records themselves, so it stays small as the store grows."""

    def __init__(self) -> None:
        self._ids: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self._places: dict[str, object] = {}
        self._order: list[str] = []

    def __len__(self) -> int:
        return len(self._places)

    @property
    def ids(self) -> tuple[str, ...]:
        """Every record's id, in the order they were remembered."""
        return tuple(self._order)

    def place(self, record_id: str) -> object | None:
        """Where the store keeps that record, or None where the index has never seen it."""
        return self._places.get(record_id)

    def add(self, record: Record, place: object) -> None:
        """Indexes the record under everything it can be looked up by."""
        if record.id not in self._places:
            self._order.append(record.id)
        self._places[record.id] = place
        for field, words in self._words(record):
            for word in words:
                self._ids[field][word.casefold()].add(record.id)

    def remove(self, record: Record) -> None:
        """Takes the record out of the index; the store keeps its own copy."""
        self._places.pop(record.id, None)
        if record.id in self._order:
            self._order.remove(record.id)
        for field, words in self._words(record):
            for word in words:
                self._ids[field][word.casefold()].discard(record.id)

    def find(
        self,
        subject: str | None = None,
        name: str | None = None,
        keyword: str | None = None,
        claim: str | None = None,
        told: str | None = None,
        source: str | None = None,
    ) -> tuple[str, ...]:
        """The ids of the records answering every field named, in the order they were remembered; every id where no
        field is named."""
        asked = [
            (field, word)
            for field, word in ((SUBJECT, subject), (NAME, name), (KEYWORD, keyword), (CLAIM, claim), (TOLD, told), (SOURCE, source))
            if word is not None
        ]
        if not asked:
            return tuple(self._order)
        found: set[str] | None = None
        for field, word in asked:
            matching = set(self._ids[field].get(word.casefold(), ()))
            found = matching if found is None else found & matching
            if not found:
                return ()
        return tuple(record_id for record_id in self._order if found is not None and record_id in found)

    def under(self, word: str) -> tuple[str, ...]:
        """The ids of the records a subject, a name or a keyword leads to: what attending to a word brings up."""
        folded = word.casefold()
        found = self._ids[SUBJECT].get(folded, set()) | self._ids[NAME].get(folded, set()) | self._ids[KEYWORD].get(folded, set())
        return tuple(record_id for record_id in self._order if record_id in found)

    def _words(self, record: Record) -> tuple[tuple[str, tuple[str, ...]], ...]:
        provenance = record.provenance
        return (
            (SUBJECT, record.subjects),
            (NAME, record.names),
            (KEYWORD, record.keywords),
            (CLAIM, () if record.claim is None else (record.claim.key,)),
            (TOLD, () if provenance.told is None else (provenance.told,)),
            (SOURCE, (provenance.source,)),
        )
