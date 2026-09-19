from pathlib import Path

from openmind.knowledge.builder.knowledge_base_builder import KnowledgeBaseBuilder
from openmind.knowledge.model.store import Store
from openmind.knowledge.service.knowledge_base import KnowledgeBase


def create_knowledge_base(
    domain: str, directory: Path | str | None = None, stores: dict[str, Store] | None = None
) -> KnowledgeBase:
    """A knowledge base for a domain, its knowledge kept as JSON lines under the knowledge directory, or in the stores
    given instead, by kind: experiences, beliefs, opinions, tasks, contexts, mechanisms, rules or rulesets."""
    builder = KnowledgeBaseBuilder()
    if directory is not None:
        builder.with_directory(directory)
    for kind, store in (stores or {}).items():
        builder.with_store(kind, store)
    return builder.build(domain)
