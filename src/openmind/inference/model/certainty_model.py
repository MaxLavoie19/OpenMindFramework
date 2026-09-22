from typing import Protocol

from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.service.knowledge_base import KnowledgeBase


class CertaintyModel(Protocol):
    """One way to turn a belief's justified evidence into its value, certainty and precision. `name` is how logs and
    tasks call it."""

    name: str

    def fits(self, knowledge: KnowledgeBase, belief: Belief, evidence: tuple[Evidence, ...]) -> bool: ...

    def assess(self, knowledge: KnowledgeBase, belief: Belief, evidence: tuple[Evidence, ...]) -> Belief: ...
