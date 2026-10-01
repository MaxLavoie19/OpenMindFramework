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

    def __getstate__(self) -> dict[str, object]:
        """What has been worked out stays behind when a node is copied to another process.

        **A feature can hold functions, so a node that has been read cannot be sent.** `RuleHeuristic` keeps
        the consequence library's names as a feature, and those names are closures over this process's library
        — `win chance`, `wins` and `near` are lambdas. So a node is picklable until something values it and
        never afterwards, which is the worst shape a thing can have: it crosses in a test and fails in a run.
        The same reason `ConsequenceLibrary.__getstate__` leaves its lookups behind, and the same answer.

        Measured: a judging died on `Can't pickle local object 'ConsequenceLibrary.names.<locals>.<lambda>'`
        after the teller valued the decisions in this process and the payoff judging tried to send those same
        decisions to six workers. It had been unreachable until a pool grew past what the teller narrows to.

        Nothing is lost but the work: a feature is extracted on demand, so a worker receiving a bare node
        extracts what it needs. It pays again for what this process already worked out, which is a cost to
        measure rather than a reason to send functions over a pipe."""
        return {"state": self.state, "game": self.game}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.state = state["state"]  # type: ignore[assignment]
        self.game = state["game"]
        self.features = {}
