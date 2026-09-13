from dataclasses import dataclass

from openmind.predictor.model.branch import Branch


@dataclass(frozen=True, slots=True)
class Transition:
    """What performing an action does: its possible branches."""

    action: str
    branches: tuple[Branch, ...]
