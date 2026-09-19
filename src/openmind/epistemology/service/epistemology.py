import logging
from collections.abc import Mapping

from openmind.epistemology.model.conflict import Conflict
from openmind.epistemology.service.certainty_assessor import CertaintyAssessor
from openmind.epistemology.service.coherence_checker import CoherenceChecker
from openmind.epistemology.service.justifier import Justifier
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.task import Task
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class Epistemology:
    """Makes a context's beliefs rigorous, by foundherentism: traces each belief's support back to its anchors, assesses
    its certainty from the evidence that reaches them, and finds where knowledge doesn't cohere.

    Whether a context's beliefs are reviewed as soon as they are set or later, as a task, is a setting per context until
    the time management policy decides it. A belief no certainty model can assess stays as given, and the need for a
    way to assess that kind of belief becomes a task, once per kind and context; the agent lives with it meanwhile."""

    def __init__(
        self,
        justifier: Justifier,
        certainty_assessor: CertaintyAssessor,
        coherence_checker: CoherenceChecker,
        immediate: Mapping[str, bool] | None = None,
    ) -> None:
        self._justifier = justifier
        self._assessor = certainty_assessor
        self._coherence = coherence_checker
        self._immediate = dict(immediate or {})
        self._needs: set[tuple[str, str]] = set()

    def immediate(self, context: str) -> bool:
        """Whether beliefs in that context are reviewed as soon as they are set; True where the context has no setting."""
        return self._immediate.get(context, True)

    def review(self, knowledge: KnowledgeBase, belief: Belief, need_value: tuple[Belief, ...] = (), need_time: Belief | None = None) -> Belief:
        """The belief kept, its certainty assessed from its justified evidence. Where no model fits, it is kept as given,
        and a task to find a way to assess that kind of belief is added once, worth `need_value` and expected to take
        `need_time` when they are given."""
        justification = self._justifier.justify(knowledge, belief)
        if not self._assessor.fitted(knowledge, belief, justification):
            self._need(knowledge, belief, need_value, need_time)
        return knowledge.believe(self._assessor.assess(knowledge, belief, justification))

    def audit(
        self, knowledge: KnowledgeBase, context: str, value: tuple[Belief, ...], expected_time: Belief
    ) -> tuple[Conflict, ...]:
        """Finds the context's conflicts, each logged as a warning and made a task worth that value and expected to take
        that long, as whoever audits says."""
        conflicts = self._coherence.conflicts(knowledge, context)
        for conflict in conflicts:
            self._coherence.as_task(knowledge, conflict, value, expected_time)
        return conflicts

    def _need(self, knowledge: KnowledgeBase, belief: Belief, value: tuple[Belief, ...], time: Belief | None) -> None:
        kind = self._assessor.unassessed(belief)
        if (kind, belief.context) in self._needs:
            return
        self._needs.add((kind, belief.context))
        logger.warning("No certainty model fits %s in %s: kept as given", kind, knowledge.readable_context(belief.context))
        if time is None:
            return
        knowledge.task(
            Task(
                f"find a way to assess {kind}",
                belief.context,
                value,
                time,
                tags=(("keyword", "need"), ("kind", kind)),
            )
        )
