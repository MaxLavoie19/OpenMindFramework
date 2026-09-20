from typing import Protocol

from openmind.heuristic.model.node import Node
from openmind.knowledge.model.policy import Policy
from openmind.world.model.action import Action


class Optimizer[Model](Protocol):
    """The optimizing task: the next best step from this node for the policy's sub-goal, without listing the
    alternatives, or None where the model has no solution here.

    Optimizing is the alternative to expanding. Expanding lists the valid actions and rates them; an optimizer computes
    the one that serves the goal, as an artillery piece calculates its firing solution rather than enumerating every
    ballistic one. It can be equations, a controller, or the constraint solver working the values out one at a time.

    A computed action isn't legal by construction, so what an optimizer gives is checked against the game's
    constraints, unless the optimizer is the solver itself. Model predictive control's plan is the series of steps the
    optimizers give as the search explores; an optimizer answers for one step."""

    def optimize(self, model: Model, policy: Policy, node: Node, player: str) -> Action | None: ...
