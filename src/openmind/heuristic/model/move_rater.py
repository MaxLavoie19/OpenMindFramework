from typing import Protocol

from openmind.heuristic.model.node import Node
from openmind.world.model.action import Action


class MoveRater[Model](Protocol):
    """The move value task: what each action is worth to the player taking it, in the actions' order, None for an
    action the model knows nothing about. A stateless service fills it, given the model it runs.

    **A node and not a bare state**, for the reason `PositionValuer` gives: what a position holds is worked out
    once and shared by every model asked about it. This said `state` while everything passed a node."""

    def rate(
        self, model: Model, node: Node, actions: tuple[Action, ...], player: str
    ) -> tuple[float | None, ...]: ...
