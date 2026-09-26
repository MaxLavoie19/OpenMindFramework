from typing import Protocol

from openmind.heuristic.model.node import Node


class PositionValuer[Model](Protocol):
    """The position value task: what a state is worth to each player, in the players' order, or None where the model
    knows nothing of it. A stateless service fills it, given the model it runs: a position value ruleset through the
    RBS, a lookup table, a network, …

    **A node and not a bare state**, so that what has been worked out about a position is worked out once. A
    rule-based heuristic reads a board's tactics and win chances; a network may read none of it and the node costs
    it nothing. This said `state` while every service filling it and every search calling it passed a node — so
    the structural check this protocol exists to make was not being made."""

    def values(self, model: Model, node: Node) -> tuple[float, ...] | None: ...
