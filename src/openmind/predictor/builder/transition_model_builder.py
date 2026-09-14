import math
from typing import Self

from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.predictor.model.transition_model import TransitionModel
from openmind.rule.model.python_rule import PythonRule


class TransitionModelBuilder:
    """Collects transitions, and the definitions their effects see, into a transition model."""

    def __init__(self) -> None:
        self._transitions: list[Transition] = []
        self._definitions: PythonRule | None = None

    def with_transition(self, action: str, branches: tuple[Branch, ...]) -> Self:
        if any(transition.action == action for transition in self._transitions):
            raise ValueError(f"Action {action!r} already has a transition")
        total = math.fsum(branch.probability for branch in branches)
        if not math.isclose(total, 1.0):
            raise ValueError(f"Branch probabilities of {action!r} sum to {total}, not 1")
        self._transitions.append(Transition(action, branches))
        return self

    def with_definitions(self, definitions: PythonRule) -> Self:
        self._definitions = definitions
        return self

    def build(self) -> TransitionModel:
        return TransitionModel(tuple(self._transitions), self._definitions)
