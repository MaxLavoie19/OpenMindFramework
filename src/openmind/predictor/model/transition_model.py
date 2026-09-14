from dataclasses import dataclass

from openmind.predictor.model.transition import Transition
from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class TransitionModel:
    """Every transition of a domain, and the definitions whose names their effects see (None for none)."""

    transitions: tuple[Transition, ...]
    definitions: PythonRule | None = None
