from typing import Protocol

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy


class PolicyValuer[Model](Protocol):
    """The policy value task: what each policy is worth following in this node, in the policies' order, None for a
    policy the model knows nothing about. A policy takes the utility of the move it decided on, and that utility is
    what trains a model of this task."""

    def rate(
        self, model: Model, node: Node, policies: tuple[Policy, ...], player: str
    ) -> tuple[float | None, ...]: ...
