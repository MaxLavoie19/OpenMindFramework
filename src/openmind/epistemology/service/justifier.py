import logging

from openmind.epistemology.model.justification import Justification
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class Justifier:
    """Traces a belief's support back to its anchors, following every source's `rests_on`.

    Direct experiences are anchors, and so are frozen rules: what the agent received and what an application declared.
    A belief, an opinion or a rule OMF produced supports only as far as what it rests on does, so support is followed
    through them; an id met again on its own path is circular, and gives no support. A piece of evidence is justified
    when its path reaches an anchor."""

    def justify(self, knowledge: KnowledgeBase, belief: Belief) -> Justification:
        anchors: dict[str, None] = {}
        circular: dict[str, None] = {}
        justified: list[int] = []
        reached: set[frozenset[str]] = set()
        for index, evidence in enumerate(belief.evidence):
            found: dict[str, None] = {}
            for entry in evidence.source.rests_on:
                self._trace(knowledge, entry, (belief.id,), found, circular)
            if found:
                justified.append(index)
                reached.add(frozenset(found))
                anchors.update(found)
        justification = Justification(belief.id, tuple(anchors), tuple(circular), len(reached), tuple(justified))
        logger.debug(
            "%s rests on %d anchors through %d of its %d pieces of evidence%s",
            belief.variable,
            len(justification.anchors),
            len(justified),
            len(belief.evidence),
            f"; circular through {', '.join(justification.circular)}" if justification.circular else "",
        )
        return justification

    def _trace(
        self,
        knowledge: KnowledgeBase,
        entry: str,
        path: tuple[str, ...],
        anchors: dict[str, None],
        circular: dict[str, None],
    ) -> None:
        if entry in path:
            circular[entry] = None
            return
        if knowledge.experienced(entry) is not None:
            anchors[entry] = None
            return
        rule = knowledge.rule(entry)
        if rule is not None:
            if knowledge.frozen(rule):
                anchors[entry] = None
            else:
                for further in rule.source.rests_on:
                    self._trace(knowledge, further, (*path, entry), anchors, circular)
            return
        believed = knowledge.belief_by_id(entry)
        if believed is not None:
            for evidence in believed.evidence:
                for further in evidence.source.rests_on:
                    self._trace(knowledge, further, (*path, entry), anchors, circular)
            return
        held = next((opinion for opinion in knowledge.opinions() if opinion.id == entry), None)
        if held is not None:
            for reason in held.reasons:
                for further in reason.rests_on:
                    self._trace(knowledge, further, (*path, entry), anchors, circular)
