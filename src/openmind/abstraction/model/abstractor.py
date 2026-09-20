from typing import Protocol

from openmind.heuristic.model.node import Node
from openmind.world.model.state import State


class Abstractor[Model](Protocol):
    """The abstraction task: the state as one level of the hierarchy sees it — the detail that level works on, and none
    of the rest.

    Every level abstracts from what the world holds, each in its own way: macromanagement sees the economy and the
    army, micromanagement the units in this fight, and a coding agent's high level a file's path and not its content.
    What the level doesn't see, it fetches by acting: reading the file is a move of its own game.

    It is given a node, so an abstraction reads whatever features have already been extracted, and answers the state
    the level holds; None where the model has nothing to say about this state, and the level then sees it as it is.
    A stateless service fills it, given the model it runs: an abstraction ruleset through the RBS, a decoder, a
    summary, a relaxation that drops detail."""

    def abstract(self, model: Model, node: Node, context: str) -> State | None: ...
