import logging

from collections.abc import Sequence

from openmind.inference.constant.certainty_constant import AGAINST_AN_ANCHOR, CANNOT_BOTH_HOLD, VALUES_DISAGREE
from openmind.inference.model.conflict import Conflict
from openmind.inference.model.inference_budget import InferenceBudget
from openmind.inference.service.forward_chainer import ForwardChainer
from openmind.inference.service.justifier import Justifier
from openmind.knowledge.constant.knowledge_constant import CONFLICT
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.identifier import new_identifier
from openmind.knowledge.model.task import Task
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.statement.model.clause import Clause

logger = logging.getLogger(__name__)


class CoherenceChecker:
    """Finds where a context's knowledge doesn't cohere, and makes each conflict a warning and a task, since a model may
    have solved a problem incorrectly and the proper inference may be something else.

    Three conflicts are found, and the third is what being inside the engine buys.

    - A belief whose justified evidence names **different values**.
    - A belief whose value **a direct experience it rests on** contradicts.
    - **Clauses that resolve to the contradiction.** Two rules can be jointly impossible without either being about
      the same named variable as the other, and comparing values can never see it: that a thing reaches fourteen
      and that nothing reaches more than ten are statements about different variables, and they cannot both hold.
      Finding that is one resolution step, which is why this belongs here and not where it was.

    A conflict is not resolved. It becomes a warning and a task, because which of the two to give up is not a thing
    the checker can know."""

    def __init__(self, justifier: Justifier, forward_chainer: ForwardChainer | None = None) -> None:
        self._justifier = justifier
        self._chainer = ForwardChainer() if forward_chainer is None else forward_chainer

    def conflicts(self, knowledge: KnowledgeBase, context: str) -> tuple[Conflict, ...]:
        found: list[Conflict] = []
        for belief in knowledge.beliefs(context):
            found.extend(self._of(knowledge, belief))
        for conflict in found:
            logger.warning(
                "Conflict in %s on %s: %s, between %s",
                knowledge.readable_context(conflict.context),
                conflict.variable,
                conflict.kind,
                ", ".join(conflict.ids),
            )
        return tuple(found)

    def as_task(self, knowledge: KnowledgeBase, conflict: Conflict, value: tuple[Belief, ...], expected_time: Belief) -> Task:
        """The task of investigating the conflict, worth what its finder says it is and expected to take as long."""
        return knowledge.task(
            Task(
                f"investigate the conflict on {conflict.variable}: {conflict.kind}",
                conflict.context,
                value,
                expected_time,
                tags=(("keyword", "conflict"), ("conflict", conflict.id), *(("about", entry) for entry in conflict.ids)),
            )
        )

    def contradictory(
        self, clauses: Sequence[Clause], context: str, budget: InferenceBudget
    ) -> tuple[Conflict, ...]:
        """Whether those clauses can all hold at once, and which of them cannot.

        Chained rather than asked. A question is answered by denying it and driving at a contradiction, and here
        the contradiction *is* the question — there is nothing to deny — so what is wanted is to put the clauses
        together and see whether the contradiction falls out.

        What comes back names the clauses each contradiction leaned on, since those are what a person has to
        choose between. The whole set is never the answer, and neither is whichever one happened to be added
        last."""
        found = tuple(
            Conflict(
                new_identifier(CONFLICT),
                ", ".join(one.rests_on) or "clauses",
                context,
                one.rests_on,
                CANNOT_BOTH_HOLD,
            )
            for one in self._chainer.chain(clauses, budget)
            if one.conclusion.empty
        )
        for conflict in found:
            logger.warning(
                "Conflict in %s: %s, between %s",
                context,
                conflict.kind,
                ", ".join(conflict.ids) or "clauses with no name",
            )
        return found

    def _of(self, knowledge: KnowledgeBase, belief: Belief) -> list[Conflict]:
        justification = self._justifier.justify(knowledge, belief)
        justified = [belief.evidence[index] for index in justification.justified]
        conflicts: list[Conflict] = []
        if len({evidence.value for evidence in justified}) > 1:
            conflicts.append(Conflict(new_identifier(CONFLICT), belief.variable, belief.context, (belief.id,), VALUES_DISAGREE))
        for evidence in justified:
            for entry in evidence.source.rests_on:
                experience = knowledge.experienced(entry)
                if experience is not None and experience.variable == belief.variable and experience.value != belief.value:
                    conflicts.append(
                        Conflict(new_identifier(CONFLICT), belief.variable, belief.context, (belief.id, entry), AGAINST_AN_ANCHOR)
                    )
        return conflicts
