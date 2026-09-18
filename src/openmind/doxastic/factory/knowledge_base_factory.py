from pathlib import Path

from openmind.doxastic.builder.knowledge_base_builder import KnowledgeBaseBuilder
from openmind.doxastic.model.record_store import RecordStore
from openmind.doxastic.model.rule_store import RuleStore
from openmind.doxastic.service.knowledge_base import KnowledgeBase


def create_knowledge_base(
    domain: str,
    directory: Path | str | None = None,
    record_store: RecordStore | None = None,
    rule_store: RuleStore | None = None,
) -> KnowledgeBase:
    """A knowledge base for a domain, its records and rules kept as JSON lines under the knowledge directory, or in the
    stores given instead."""
    builder = KnowledgeBaseBuilder()
    if directory is not None:
        builder.with_directory(directory)
    if record_store is not None:
        builder.with_store(record_store)
    if rule_store is not None:
        builder.with_rule_store(rule_store)
    return builder.build(domain)
