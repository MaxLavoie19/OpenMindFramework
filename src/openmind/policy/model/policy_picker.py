from typing import Protocol

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy


class PolicyPicker[Model](Protocol):
    """The policy picking task: which policies are worth expanding here, best first, and which aren't worth
    considering at all.

    Picking at random is rarely worth expanding; when it is — when there is no time to think — nothing else is worth
    considering."""

    def pick(self, model: Model, node: Node, policies: tuple[Policy, ...], player: str) -> tuple[Policy, ...]: ...
