import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from openmind.inference.model.fact import Fact
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.model.source import Source
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)

#: What produced these, recorded so a conclusion can say how it was reached.
INFERENCE = "inference"
#: How a fact is named as a variable: what is being said, of what.
NAMED = "{kind} {about}"


class FactRecorder:
    """Writes down what was concluded, so that it outlives the run that concluded it.

    Everything OMF works out about a game has so far existed only while the program that worked it out was running:
    how far a piece reaches, what one thing takes in, what a move threatens. Written down, a conclusion can be used
    by something that did not compute it, and can be argued with — which is the point of keeping what it rests on.

    A fact becomes a belief in the game's context, holding the number it concluded, with a source naming inference
    as the mechanism and listing the beliefs and rules it was drawn from. A chain therefore survives as a chain:
    a queen is worth at least fourteen because a queen takes in a rook and a rook reaches fourteen, and each of
    those is a belief of its own that this one points at.

    Nothing here decides how sure to be. A conclusion drawn from rules the game declared is as good as the rules; one
    drawn from rules OMF induced is as good as those, which is a question about the induction and not about the
    reasoning — so certainty is what the caller says it is, and the default says the reasoning added no doubt of its
    own."""

    def record(
        self,
        knowledge_base: KnowledgeBase,
        context: str,
        facts: Sequence[Fact],
        certainty: float = 1.0,
        at: datetime | None = None,
    ) -> tuple[Belief, ...]:
        """Each of those facts as a belief, with what it rests on pointing at the beliefs it was drawn from."""
        mechanism = knowledge_base.ensure_mechanism(INFERENCE)
        known = knowledge_base.ensure_context(context)
        when = at or datetime.now(UTC)
        kept: dict[str, Belief] = {}
        for fact in self._ordered(facts):
            named = NAMED.format(kind=fact.kind, about=" ".join(str(one) for one in fact.about))
            rests_on = tuple(
                kept[self._named(one)].id for one in fact.from_facts if self._named(one) in kept
            )
            source = Source(
                mechanism.id,
                (("rules", len(fact.from_rules)), ("facts", len(fact.from_facts))),
                when,
                rests_on,
            )
            belief = knowledge_base.believe(
                Belief(
                    named,
                    known.id,
                    fact.held,
                    certainty=certainty,
                    evidence=(Evidence(fact.held, certainty, source),),
                )
            )
            kept[named] = belief
        logger.info("Wrote down %d of %d conclusions about %s", len(kept), len(facts), context)
        return tuple(kept.values())

    def _ordered(self, facts: Sequence[Fact]) -> list[Fact]:
        """The facts with what they rest on first, so a conclusion can point at its premises rather than at nothing.

        A chain written down out of order loses what makes it a chain."""
        found: list[Fact] = []
        seen: set[str] = set()

        def put(fact: Fact) -> None:
            named = self._named(fact)
            if named in seen:
                return
            seen.add(named)
            for one in fact.from_facts:
                put(one)
            found.append(fact)

        for fact in facts:
            put(fact)
        return found

    def _named(self, fact: Fact) -> str:
        return NAMED.format(kind=fact.kind, about=" ".join(str(one) for one in fact.about))
