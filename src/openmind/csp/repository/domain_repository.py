from collections.abc import Iterable

from openmind.csp.model.wipeout import Wipeout
from openmind.structure.model.value import Value


class DomainRepository:
    """The values every variable has left, and the trail of what propagation took away.

    **A search node costs what it changed, not what exists.** Copying every domain to make a node is what a search
    over a few dozen variables can afford and a search over a hundred thousand cannot; a trail replaces the copy
    with a mark, and backtracking with putting back the few values removed since it.

    It is working storage for one search, created by the search and handed to the propagators, which keep nothing
    themselves. The same values are removed and restored in place, so nothing that reads a domain may hold on to
    it past the call it was read in."""

    __slots__ = ("values", "_removed")

    def __init__(self, variables: Iterable[tuple[str, Iterable[Value]]]) -> None:
        self.values: dict[str, set[Value]] = {name: set(values) for name, values in variables}
        self._removed: list[tuple[str, Value]] = []

    @property
    def height(self) -> int:
        """How many removals have been recorded, which is the mark to unwind back to."""
        return len(self._removed)

    def remove(self, variable: str, value: Value) -> bool:
        """Takes the value out of the variable's domain, recording it, and says whether it was there to take.
        Raises Wipeout where that leaves the variable with nothing."""
        domain = self.values[variable]
        if value not in domain:
            return False
        domain.discard(value)
        self._removed.append((variable, value))
        if not domain:
            raise Wipeout(variable)
        return True

    def fix(self, variable: str, value: Value) -> None:
        """Removes every other value from the variable's domain, recording each. The value itself is not checked
        against the domain: a search assigns what the domain offered it."""
        for other in [held for held in self.values[variable] if held != value]:
            self.remove(variable, other)

    def undo_to(self, height: int) -> tuple[str, ...]:
        """Puts back every value recorded since that height, most recent first, and says which variables grew, each
        once."""
        restored: dict[str, None] = {}
        while len(self._removed) > height:
            variable, value = self._removed.pop()
            self.values[variable].add(value)
            restored[variable] = None
        return tuple(restored)

    def changed_since(self, height: int) -> tuple[str, ...]:
        """The variables something was taken from since that height, each once, most recent first."""
        changed: dict[str, None] = {}
        for index in range(len(self._removed) - 1, height - 1, -1):
            changed[self._removed[index][0]] = None
        return tuple(changed)

    def settled(self, variable: str) -> bool:
        """Whether the variable has one value left, which a search treats as assigned."""
        return len(self.values[variable]) == 1

    def only(self, variable: str) -> Value:
        """The variable's one remaining value; raises StopIteration where it has more than one or none."""
        return next(iter(self.values[variable]))
