import logging

from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class Foundherentism:
    """What OMF's support rests on, in the end.

    This is the doctrine, and it is the whole of what `epistemology` keeps back from the engine. Two kinds of thing
    are taken as anchors and nothing else is:

    - **a direct experience** — what the agent actually received, kept word for word;
    - **a rule an application declared and did not declare open** — what the programmer said the world is like.

    Everything else supports only as far as what it rests on does, and a circle of claims gets no support from
    being a circle.

    It is foundherentist rather than foundationalist because the anchors are not certain and are not many: they are
    where the tracing stops, not where doubt stops. A frozen rule an application got wrong stays an anchor, and
    what contradicts it becomes a warning rather than a revision — which is a decision about who is responsible
    for a game's rules, not a claim that the rules are true.

    It keeps nothing: built once and asked about an entry at a time."""

    def anchor(self, knowledge: KnowledgeBase, entry_id: str) -> bool:
        """Whether support that reaches that entry has reached ground."""
        if knowledge.experienced(entry_id) is not None:
            return True
        rule = knowledge.rule(entry_id)
        return rule is not None and knowledge.frozen(rule)
