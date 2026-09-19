import logging
from collections.abc import Sequence
from dataclasses import replace

from openmind.epistemology.model.certainty_model import CertaintyModel
from openmind.epistemology.model.justification import Justification
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.service.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class CertaintyAssessor:
    """Makes a belief's certainty as rigorous as its evidence allows, with the first certainty model that fits it. Only
    evidence whose justification reaches an anchor counts. A belief without justified evidence, or one no model fits,
    stays as given: the agent lives with it, and `unassessed` says what kind of belief it had no way to assess."""

    def __init__(self, models: Sequence[CertaintyModel]) -> None:
        self._models = tuple(models)

    def assess(self, knowledge: KnowledgeBase, belief: Belief, justification: Justification) -> Belief:
        evidence = tuple(belief.evidence[index] for index in justification.justified)
        if not evidence:
            return belief
        for model in self._models:
            if model.fits(knowledge, belief, evidence):
                assessed = model.assess(knowledge, belief, evidence)
                logger.info(
                    "%s in %s: %r at %.3g by the %s model, from %d of %d pieces of evidence",
                    belief.variable,
                    knowledge.readable_context(belief.context),
                    assessed.value,
                    assessed.certainty,
                    model.name,
                    len(evidence),
                    len(belief.evidence),
                )
                return replace(assessed, evidence=belief.evidence)
        return belief

    def fitted(self, knowledge: KnowledgeBase, belief: Belief, justification: Justification) -> bool:
        """Whether a model fits the belief's justified evidence; True where there is none to assess."""
        evidence = tuple(belief.evidence[index] for index in justification.justified)
        return not evidence or any(model.fits(knowledge, belief, evidence) for model in self._models)

    def unassessed(self, belief: Belief) -> str:
        """The kind of belief no model could assess, as a task names it."""
        return f"{type(belief.value).__name__} beliefs"
