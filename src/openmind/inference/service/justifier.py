import logging

from openmind.inference.model.anchoring import Anchoring
from openmind.inference.model.derivation import Derivation
from openmind.inference.model.justification import Justification
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class Justifier:
    """Traces what something rests on, back to whatever the caller counts as an anchor.

    A belief, an opinion or a rule OMF produced supports only as far as what it rests on does, so support is
    followed through them. An id met again on its own path is circular and gives no support: a thing cannot hold
    itself up, however many steps it takes to come back round. A piece of evidence is justified when its path
    reaches an anchor.

    **What counts as an anchor is asked, not assumed.** Following `rests_on` and noticing where a path bends back
    on itself is mechanical, and deciding where to stop is a position about knowledge — so the position is
    injected, and the tracing does not hold one. Foundherentism says a direct experience and a rule an application
    declared are anchors; the tracing would not change a line if something else were said.

    It traces a derivation the same way, because a derivation and a belief's evidence are the same shape: a thing
    with things under it, and things under those. What is counted for both is how many *distinct* sets of anchors
    the support reaches — which is not decoration. Two reasons for one conclusion are two reasons only where they
    reach different ground; two proofs leaning on the same doubtful clause are one reason wearing two hats, and
    reading them as two is how a conclusion comes to look better supported than it is.

    It keeps nothing: built once with what it counts as an anchor, it is given what to trace on every call."""

    def __init__(self, anchoring: Anchoring) -> None:
        self._anchoring = anchoring

    def justify(self, knowledge: KnowledgeBase, belief: Belief) -> Justification:
        """What that belief rests on: the anchors reached, the ids met twice, and which evidence got anywhere."""
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

    def independent(self, derivations: tuple[Derivation, ...]) -> int:
        """How many of those derivations are reasons of their own.

        Derivations resting on exactly the same clauses are one reason found more than once. This is the same
        counting `justify` does over evidence, asked of proofs instead."""
        return len({frozenset(one.rests_on) for one in derivations if one.rests_on})

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
        if self._anchoring.anchor(knowledge, entry):
            anchors[entry] = None
            return
        for further in self._rests_on(knowledge, entry):
            self._trace(knowledge, further, (*path, entry), anchors, circular)

    def _rests_on(self, knowledge: KnowledgeBase, entry: str) -> tuple[str, ...]:
        """What that entry was drawn from, whatever kind of entry it is."""
        rule = knowledge.rule(entry)
        if rule is not None:
            return tuple(rule.source.rests_on)
        believed = knowledge.belief_by_id(entry)
        if believed is not None:
            return tuple(further for evidence in believed.evidence for further in evidence.source.rests_on)
        held = next((opinion for opinion in knowledge.opinions() if opinion.id == entry), None)
        if held is not None:
            return tuple(further for reason in held.reasons for further in reason.rests_on)
        return ()
