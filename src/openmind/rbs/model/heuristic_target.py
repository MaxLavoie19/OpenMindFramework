from dataclasses import dataclass

from openmind.knowledge.service.knowledge_base import KnowledgeBase


@dataclass(frozen=True, slots=True)
class HeuristicTarget:
    """Where a producer of heuristic rules links what it fits: the position value ruleset of that context, named as
    applications name it, in that knowledge base."""

    knowledge_base: KnowledgeBase
    context: str
