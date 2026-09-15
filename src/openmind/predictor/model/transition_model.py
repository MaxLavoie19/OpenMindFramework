from dataclasses import dataclass

from openmind.predictor.model.branch import Branch
from openmind.predictor.model.transition import Transition
from openmind.rule.model.python_rule import PythonRule


@dataclass(frozen=True, slots=True)
class TransitionModel:
    """Every transition of a domain, the definitions whose names their effects see (None for none), and, in a domain
    where players act at once, the resolution: branches whose effects run after every player's action, reading the
    combined choices (None for none)."""

    transitions: tuple[Transition, ...]
    definitions: PythonRule | None = None
    resolution: tuple[Branch, ...] | None = None
