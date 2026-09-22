from typing import Protocol

from openmind.knowledge.service.knowledge_base import KnowledgeBase


class Anchoring(Protocol):
    """What counts as something support can rest on.

    Tracing what a conclusion rests on is mechanical: follow what each thing was drawn from until there is nothing
    further to follow, and notice where the path comes back on itself. Deciding *where to stop* is not mechanical
    at all. It is a position about what knowledge is founded on, and holding one is `epistemology`'s job, not the
    engine's.

    So the engine asks. A stateless service fills it — foundherentism answers that a direct experience and a rule
    an application declared are anchors, and something else could answer otherwise without a line of the tracing
    changing."""

    def anchor(self, knowledge: KnowledgeBase, entry_id: str) -> bool: ...
