from dataclasses import dataclass

from openmind.predictor.model.effect import Effect


@dataclass(frozen=True, slots=True)
class Branch:
    """One possible outcome of an action: its probability and the effects that produce it."""

    probability: float
    effects: tuple[Effect, ...]
