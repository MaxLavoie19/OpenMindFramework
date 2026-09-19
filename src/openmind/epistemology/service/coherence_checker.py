import logging

from openmind.epistemology.constant.epistemology_constant import AGAINST_AN_ANCHOR, VALUES_DISAGREE
from openmind.epistemology.model.conflict import Conflict
from openmind.epistemology.service.justifier import Justifier
from openmind.knowledge.constant.knowledge_constant import CONFLICT
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.identifier import new_identifier
from openmind.knowledge.model.task import Task
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class CoherenceChecker:
    """Finds where a context's knowledge doesn't cohere, and makes each conflict a warning and a task, since a model may
    have solved a problem incorrectly and the proper inference may be something else.

    Two conflicts are found: a belief whose justified evidence names different values, and a belief whose value a
    direct experience it rests on contradicts."""

    def __init__(self, justifier: Justifier) -> None:
        self._justifier = justifier

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
