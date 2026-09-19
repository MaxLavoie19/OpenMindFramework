from collections.abc import Callable
from dataclasses import dataclass, field

from openmind.world.model.state import State


@dataclass(slots=True)
class Node:
    """A state with what has been worked out about it: the game it is in, and the features extracted from it.

    A feature is extracted by the first model that asks for it and kept, so every model after shares it: `feature` is a
    memoized call. A model that reads no feature, such as a network, pays nothing for the node. A search builds one per
    state it explores, and anything else valuing a position builds one too.

    This is the one model here that isn't frozen: what is worked out about a state is added as it is worked out.

    `game` is what a feature is extracted through. It is the temporary `RuleBasedGame` facade for now; it becomes the
    simulation service and its RBS once the search and the inference readings are reworked."""

    state: State
    game: object = None
    features: dict[str, object] = field(default_factory=dict)

    def feature(self, name: str, extract: Callable[[], object]) -> object:
        """What the feature is here, extracted the first time it is asked for and shared from then on."""
        if name not in self.features:
            self.features[name] = extract()
        return self.features[name]

    def of(self, state: State) -> "Node":
        """A node for another state of the same game, its features its own."""
        return Node(state, self.game)
