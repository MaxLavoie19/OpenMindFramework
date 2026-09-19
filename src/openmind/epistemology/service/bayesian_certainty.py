import logging
from collections.abc import Mapping
from dataclasses import replace

from openmind.epistemology.constant.epistemology_constant import ANYTHING_ELSE
from openmind.epistemology.model.error_model import ErrorModel
from openmind.epistemology.service.accuracy_scorer import AccuracyScorer
from openmind.epistemology.service.even_error_model import EvenErrorModel
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.structure.model.value import Value

logger = logging.getLogger(__name__)


class BayesianCertainty:
    """Certainty for a value that isn't a number, where every piece of evidence comes from a mechanism whose accuracy is
    known. The caller's value is the prior, at the certainty the caller gave, the rest shared among the other candidate
    values: every value the evidence names, and anything else. Each piece of evidence then updates it through its
    mechanism's error model, evenly spread where the mechanism has none of its own. The value held is the likeliest,
    and the certainty its probability."""

    name = "Bayesian"

    def __init__(self, accuracy_scorer: AccuracyScorer, error_models: Mapping[str, ErrorModel] | None = None) -> None:
        self._accuracy = accuracy_scorer
        self._error_models = dict(error_models or {})
        self._even = EvenErrorModel()

    def fits(self, knowledge: KnowledgeBase, belief: Belief, evidence: tuple[Evidence, ...]) -> bool:
        return (
            bool(evidence)
            and not _number(belief.value)
            and all(self._accuracy.accuracy(knowledge, item.source.mechanism, belief.context) is not None for item in evidence)
        )

    def assess(self, knowledge: KnowledgeBase, belief: Belief, evidence: tuple[Evidence, ...]) -> Belief:
        candidates: list[Value] = list(dict.fromkeys((belief.value, *(item.value for item in evidence), ANYTHING_ELSE)))
        others = len(candidates) - 1
        prior = {
            candidate: belief.certainty if candidate == belief.value else (1.0 - belief.certainty) / others
            for candidate in candidates
        }
        posterior = dict(prior)
        for item in evidence:
            accuracy = self._accuracy.accuracy(knowledge, item.source.mechanism, belief.context)
            model = self._error_models.get(item.source.mechanism, self._even)
            for candidate in candidates:
                posterior[candidate] *= model.likelihood(item.value, candidate, float(accuracy), len(candidates))  # type: ignore[arg-type]
        total = sum(posterior.values())
        if total <= 0.0:
            logger.warning("%s: the evidence leaves no value possible; kept as given", belief.variable)
            return belief
        chosen = max(candidates, key=lambda candidate: posterior[candidate])
        if chosen == ANYTHING_ELSE:
            chosen = max((candidate for candidate in candidates if candidate != ANYTHING_ELSE), key=lambda c: posterior[c])
        certainty = posterior[chosen] / total
        logger.debug(
            "%s: %s",
            belief.variable,
            ", ".join(f"{candidate!r} {posterior[candidate] / total:.3g}" for candidate in candidates),
        )
        return replace(belief, value=chosen, certainty=certainty)


def _number(value: Value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
