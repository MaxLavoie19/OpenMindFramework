from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.service.memory_evictor import evict_oldest
from openmind.search.model.search_node import SearchNode
from openmind.world.model.state import State


class SearchTreeCache:
    """The tree each position was explored from, built once and given to the search, which keeps nothing itself.

    A planner strategizes between moves and shouldn't throw away what it found: the next search starts from the tree
    of the position it is now in, which is a child of the last root, and what can no longer be reached goes with the
    rest of that tree. The process's memory guard evicts the oldest when memory runs short, and a copy sent to another
    process arrives empty."""

    def __init__(self) -> None:
        self._trees: dict[State, SearchNode] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __getstate__(self) -> dict[str, object]:
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        self._trees = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def tree(self, state: State) -> SearchNode | None:
        """What was explored from that position, or None where it hasn't been."""
        return self._trees.get(state)

    def keep(self, state: State, tree: SearchNode) -> SearchNode:
        """Keeps the tree explored from that position, and every position it reached, so a move played carries on from
        what was found there. Positions it can no longer reach are dropped."""
        self._memory_guard.remembered()
        self._trees = {state: tree}
        for child in tree.children.values():
            self._trees[child.node.state] = child
        return tree

    def memory_entries(self) -> int:
        return len(self._trees)

    def evict_memory(self, entries: int) -> None:
        evict_oldest(self._trees, entries)

    def clear_memory(self) -> None:
        self._trees.clear()
