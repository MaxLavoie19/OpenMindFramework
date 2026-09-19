import logging
import math

from openmind.epistemology.constant.epistemology_constant import ACCURACY, RIGHT, SCORED, SPREAD, SQUARED_ERROR
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)

#: How many cases a declared accuracy counts for, the weight of the uninformative Beta(1, 1) it stands in for.
PRIOR_CASES = 2


class AccuracyScorer:
    """Measures mechanisms against what later turns out true.

    When an anchor settles a variable in a context, every piece of evidence any belief held about it scores its
    mechanism: right or wrong for a value that isn't a number, how far off for one that is. A mechanism's accuracy is a
    belief about it in that context: the share it got right, starting from its declared accuracy (uninformative, 0.5,
    without one) as if it had been scored twice. Its spread is the root of its mean squared error on numbers."""

    def settle(self, knowledge: KnowledgeBase, variable: str, context: str, anchor_id: str) -> tuple[Belief, ...]:
        """Scores every mechanism whose evidence bore on the variable against the anchor's value; gives back the
        mechanisms' accuracy and spread beliefs as updated."""
        anchor = knowledge.experienced(anchor_id)
        if anchor is None:
            raise KeyError(f"No direct experience {anchor_id} to settle {variable} by")
        truth = anchor.value
        updated: list[Belief] = []
        for belief in knowledge.beliefs(context):
            if belief.variable != variable:
                continue
            for evidence in belief.evidence:
                if anchor_id in evidence.source.rests_on:
                    continue
                updated.append(self._score(knowledge, evidence.source.mechanism, context, evidence.value, truth, anchor_id))
        return tuple(updated)

    def accuracy(self, knowledge: KnowledgeBase, mechanism_id: str, context: str) -> float | None:
        """The mechanism's accuracy in that context: as measured, else as declared, else None: not known."""
        measured = knowledge.belief(ACCURACY.format(mechanism=mechanism_id), context)
        if measured is not None and dict(measured.tags).get(SCORED):
            return float(measured.value)  # type: ignore[arg-type]
        mechanism = knowledge.mechanism_by_id(mechanism_id)
        return None if mechanism is None else mechanism.declared_accuracy

    def spread(self, knowledge: KnowledgeBase, mechanism_id: str, context: str) -> float | None:
        """How far off the mechanism is on numbers in that context, as measured; None before it is scored on one."""
        measured = knowledge.belief(SPREAD.format(mechanism=mechanism_id), context)
        return None if measured is None else float(measured.value)  # type: ignore[arg-type]

    def _score(
        self, knowledge: KnowledgeBase, mechanism_id: str, context: str, said: Value, truth: Value, anchor_id: str
    ) -> Belief:
        name = knowledge.readable_mechanism(mechanism_id)
        if _number(said) and _number(truth):
            variable = SPREAD.format(mechanism=mechanism_id)
            held = knowledge.belief(variable, context)
            tags = dict(held.tags) if held is not None else {}
            scored = int(tags.get(SCORED, 0)) + 1  # type: ignore[arg-type]
            squared = float(tags.get(SQUARED_ERROR, 0.0)) + (float(said) - float(truth)) ** 2  # type: ignore[arg-type]
            spread = math.sqrt(squared / scored)
            logger.info("%s was off by %.4g: its spread is %.4g over %d", name, float(said) - float(truth), spread, scored)  # type: ignore[arg-type]
            return knowledge.believe(
                Belief(variable, context, spread, tags=((SCORED, scored), (SQUARED_ERROR, squared), ("mechanism", mechanism_id)))
            )
        variable = ACCURACY.format(mechanism=mechanism_id)
        held = knowledge.belief(variable, context)
        tags = dict(held.tags) if held is not None else {}
        scored = int(tags.get(SCORED, 0)) + 1  # type: ignore[arg-type]
        right = int(tags.get(RIGHT, 0)) + (1 if said == truth else 0)  # type: ignore[arg-type]
        mechanism = knowledge.mechanism_by_id(mechanism_id)
        declared = 0.5 if mechanism is None or mechanism.declared_accuracy is None else mechanism.declared_accuracy
        accuracy = (right + declared * PRIOR_CASES) / (scored + PRIOR_CASES)
        logger.info(
            "%s said %r where %r turned out true: %s, accuracy %.3g over %d",
            name,
            said,
            truth,
            "right" if said == truth else "wrong",
            accuracy,
            scored,
        )
        return knowledge.believe(
            Belief(
                variable,
                context,
                accuracy,
                tags=((SCORED, scored), (RIGHT, right), ("mechanism", mechanism_id), ("anchor", anchor_id)),
            )
        )


def _number(value: Value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

