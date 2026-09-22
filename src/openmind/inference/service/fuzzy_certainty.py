from dataclasses import replace

from openmind.knowledge.model.belief import Belief
from openmind.knowledge.model.evidence import Evidence
from openmind.knowledge.service.knowledge_base import KnowledgeBase
from openmind.structure.model.value import Value


class FuzzyCertainty:
    """Certainty where there are no measurements to reason with, in fuzzy logic over the evidence's strengths: a value
    is supported as much as its strongest evidence (fuzzy or), and held as much as it is supported and not opposed
    (fuzzy and, with not as 1 − the strongest evidence for any other value). The value held is the best supported."""

    name = "fuzzy"

    def fits(self, knowledge: KnowledgeBase, belief: Belief, evidence: tuple[Evidence, ...]) -> bool:
        return bool(evidence)

    def assess(self, knowledge: KnowledgeBase, belief: Belief, evidence: tuple[Evidence, ...]) -> Belief:
        support: dict[Value, float] = {}
        for item in evidence:
            support[item.value] = max(support.get(item.value, 0.0), min(max(item.strength, 0.0), 1.0))
        chosen = max(support, key=lambda value: support[value])
        opposed = max((strength for value, strength in support.items() if value != chosen), default=0.0)
        return replace(belief, value=chosen, certainty=min(support[chosen], 1.0 - opposed))
