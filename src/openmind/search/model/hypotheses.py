from typing import Protocol

from openmind.heuristic.model.node import Node
from openmind.search.model.hypothesis import Hypothesis


class Hypotheses[Model](Protocol):
    """The hypothesis task: what an agent can't see, as a probability distribution — every hypothesis with how likely
    it is, summing to 1.

    How a model arrives at one is its own business: a hidden Markov model and a ruleset express it differently, and
    both answer the same question. Empty where the model knows nothing, and for a game that hides nothing."""

    def about(self, model: Model, node: Node, agent: str) -> tuple[Hypothesis, ...]: ...
