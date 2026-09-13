from dataclasses import dataclass

from openmind.predictor.model.transition import Transition


@dataclass(frozen=True, slots=True)
class TransitionModel:
    """Every transition of a domain."""

    transitions: tuple[Transition, ...]
