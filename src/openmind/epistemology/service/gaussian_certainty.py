import math
from dataclasses import replace

from openmind.epistemology.service.accuracy_scorer import AccuracyScorer
from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.structure.model.value import Value


class GaussianCertainty:
    """Certainty for a number, where every piece of evidence comes from a mechanism whose spread on numbers has been
    measured: each estimate is a Gaussian around the value it names, as wide as its mechanism's spread, and the
    estimates are combined, each weighed by the inverse of its variance. The value held is their combined mean, the
    precision their combined spread; certainty stays as the caller gave it, since a spread says how far off, not how
    sure."""

    name = "Gaussian"

    def __init__(self, accuracy_scorer: AccuracyScorer) -> None:
        self._accuracy = accuracy_scorer

    def fits(self, knowledge: KnowledgeBase, belief: Belief, evidence: tuple[Evidence, ...]) -> bool:
        return (
            bool(evidence)
            and _number(belief.value)
            and all(_number(item.value) for item in evidence)
            and all(self._spread(knowledge, item, belief.context) is not None for item in evidence)
        )

    def assess(self, knowledge: KnowledgeBase, belief: Belief, evidence: tuple[Evidence, ...]) -> Belief:
        weights: list[tuple[float, float]] = []
        for item in evidence:
            spread = self._spread(knowledge, item, belief.context)
            variance = max(float(spread) ** 2, 1e-12)  # type: ignore[arg-type]
            weights.append((float(item.value), 1.0 / variance))  # type: ignore[arg-type]
        total = sum(weight for _, weight in weights)
        mean = sum(value * weight for value, weight in weights) / total
        return replace(belief, value=mean, precision=math.sqrt(1.0 / total))

    def _spread(self, knowledge: KnowledgeBase, item: Evidence, context: str) -> float | None:
        return self._accuracy.spread(knowledge, item.source.mechanism, context)


def _number(value: Value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
