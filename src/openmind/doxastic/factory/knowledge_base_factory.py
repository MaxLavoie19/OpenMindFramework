from pathlib import Path

from openmind.doxastic.builder.knowledge_base_builder import KnowledgeBaseBuilder
from openmind.doxastic.model.record_store import RecordStore
from openmind.doxastic.service.knowledge_base import KnowledgeBase


def create_knowledge_base(
    domain: str, directory: Path | str | None = None, record_store: RecordStore | None = None
) -> KnowledgeBase:
    """A knowledge base for a domain, its records kept as JSON lines under the knowledge directory, or in the store
    given instead."""
    builder = KnowledgeBaseBuilder()
    if directory is not None:
        builder.with_directory(directory)
    if record_store is not None:
        builder.with_store(record_store)
    return builder.build(domain)
