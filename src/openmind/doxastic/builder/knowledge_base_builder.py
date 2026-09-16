from pathlib import Path

from openmind.doxastic.constant.doxastic_constant import KNOWLEDGE_DIRECTORY, RECORDS_FILE
from openmind.doxastic.model.record_store import RecordStore
from openmind.doxastic.service.evidence_weigher import EvidenceWeigher
from openmind.doxastic.service.file_record_store import FileRecordStore
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.doxastic.service.recall_cache import RecallCache
from openmind.doxastic.service.record_index import RecordIndex
from openmind.parallel.service.memory_meter import MemoryMeter


class KnowledgeBaseBuilder:
    """Wires a knowledge base with its store, index, cache and weigher. Without a store of its own, a domain's records
    are kept as JSON lines under the knowledge directory; another store — SQLite, or anything else answering
    `RecordStore` — is given with `with_store`."""

    def __init__(self) -> None:
        self._directory = Path(KNOWLEDGE_DIRECTORY)
        self._store: RecordStore | None = None

    def with_directory(self, directory: Path | str) -> "KnowledgeBaseBuilder":
        self._directory = Path(directory)
        return self

    def with_store(self, record_store: RecordStore) -> "KnowledgeBaseBuilder":
        self._store = record_store
        return self

    def build(self, domain: str) -> KnowledgeBase:
        store = self._store if self._store is not None else FileRecordStore(self._directory / domain / RECORDS_FILE)
        return KnowledgeBase(domain, store, RecordIndex(), RecallCache(MemoryMeter()), EvidenceWeigher())
