from pathlib import Path

from openmind.doxastic.constant.doxastic_constant import KNOWLEDGE_DIRECTORY, RECORDS_FILE
from openmind.doxastic.constant.rule_kind_constant import RULES_FILE
from openmind.doxastic.model.record_store import RecordStore
from openmind.doxastic.model.rule_store import RuleStore
from openmind.doxastic.service.evidence_weigher import EvidenceWeigher
from openmind.doxastic.service.file_record_store import FileRecordStore
from openmind.doxastic.service.file_rule_store import FileRuleStore
from openmind.doxastic.service.knowledge_base import KnowledgeBase
from openmind.doxastic.service.recall_cache import RecallCache
from openmind.doxastic.service.record_index import RecordIndex
from openmind.parallel.service.memory_meter import MemoryMeter


class KnowledgeBaseBuilder:
    """Wires a knowledge base with its stores, index, cache and weigher. Without stores of its own, a domain's records
    and rules are kept as JSON lines under the knowledge directory; another store — SQLite, or anything else answering
    `RecordStore` or `RuleStore` — is given with `with_store` and `with_rule_store`."""

    def __init__(self) -> None:
        self._directory = Path(KNOWLEDGE_DIRECTORY)
        self._store: RecordStore | None = None
        self._rule_store: RuleStore | None = None

    def with_directory(self, directory: Path | str) -> "KnowledgeBaseBuilder":
        self._directory = Path(directory)
        return self

    def with_store(self, record_store: RecordStore) -> "KnowledgeBaseBuilder":
        self._store = record_store
        return self

    def with_rule_store(self, rule_store: RuleStore) -> "KnowledgeBaseBuilder":
        self._rule_store = rule_store
        return self

    def build(self, domain: str) -> KnowledgeBase:
        store = self._store if self._store is not None else FileRecordStore(self._directory / domain / RECORDS_FILE)
        rules = self._rule_store if self._rule_store is not None else FileRuleStore(self._directory / domain / RULES_FILE)
        return KnowledgeBase(domain, store, RecordIndex(), RecallCache(MemoryMeter()), EvidenceWeigher(), rules)
