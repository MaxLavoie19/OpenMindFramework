from pathlib import Path
from typing import Self

from openmind.knowledge.constant.knowledge_constant import (
    BELIEFS_FILE,
    CONTEXTS_FILE,
    EXPERIENCES_FILE,
    KNOWLEDGE_DIRECTORY,
    MECHANISMS_FILE,
    OPINIONS_FILE,
    RULES_FILE,
    GOALS_FILE,
    MODELS_FILE,
    POLICIES_FILE,
    PREFERENCES_FILE,
    RULESETS_FILE,
    TASKS_FILE,
)
from openmind.knowledge.model.store import Store
from openmind.knowledge.service.file_store import FileStore
from openmind.knowledge.service.knowledge_base import KnowledgeBase

#: The kinds of knowledge, each with the file it is kept in by default.
KINDS = {
    "experiences": EXPERIENCES_FILE,
    "beliefs": BELIEFS_FILE,
    "opinions": OPINIONS_FILE,
    "tasks": TASKS_FILE,
    "contexts": CONTEXTS_FILE,
    "mechanisms": MECHANISMS_FILE,
    "rules": RULES_FILE,
    "rulesets": RULESETS_FILE,
    "models": MODELS_FILE,
    "policies": POLICIES_FILE,
    "goals": GOALS_FILE,
    "preferences": PREFERENCES_FILE,
}


class KnowledgeBaseBuilder:
    """Wires a knowledge base with a store for each kind of knowledge. Without stores of its own, a domain's knowledge
    is kept as JSON lines under the knowledge directory; an integrator gives their own storage, kind by kind, with
    `with_store`."""

    def __init__(self) -> None:
        self._directory = Path(KNOWLEDGE_DIRECTORY)
        self._stores: dict[str, Store] = {}

    def with_directory(self, directory: Path | str) -> Self:
        self._directory = Path(directory)
        return self

    def with_store(self, kind: str, store: Store) -> Self:
        """The store for one kind of knowledge: experiences, beliefs, opinions, tasks, contexts, mechanisms, rules, rulesets, models, policies, goals or preferences."""
        if kind not in KINDS:
            raise ValueError(f"No kind of knowledge {kind!r}: the kinds are {', '.join(KINDS)}")
        self._stores[kind] = store
        return self

    def build(self, domain: str) -> KnowledgeBase:
        stores = {
            kind: self._stores.get(kind) or FileStore(self._directory / domain / file) for kind, file in KINDS.items()
        }
        return KnowledgeBase(
            domain,
            stores["experiences"],
            stores["beliefs"],
            stores["opinions"],
            stores["tasks"],
            stores["contexts"],
            stores["mechanisms"],
            stores["rules"],
            stores["rulesets"],
            stores["models"],
            stores["policies"],
            stores["goals"],
            stores["preferences"],
        )
