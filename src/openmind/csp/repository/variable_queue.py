import heapq
from collections.abc import Mapping

from openmind.csp.repository.domain_repository import DomainRepository


class VariableQueue:
    """The variables still to decide, the one with the fewest values left first.

    **The same choice the search has always made, without looking at every variable to make it.** Ties go to the
    variable in the most constraints, then to the order the parameters were declared in — and because that order
    is unique, the smallest key here is the very variable `min` over the open variables used to return.

    An entry goes stale as a domain shrinks or grows; rather than find and fix it, the queue records the size an
    entry was pushed with and drops the entry when it surfaces disagreeing with the domain. So a variable is
    pushed on every change and read at most once per push."""

    __slots__ = ("_heap", "_degree", "_position")

    def __init__(self, degree: Mapping[str, int], position: Mapping[str, int]) -> None:
        self._heap: list[tuple[int, int, int, str]] = []
        self._degree = degree
        self._position = position

    def push(self, variable: str, size: int) -> None:
        """Offers the variable at that domain size. Pushing one already there is how a change is recorded."""
        heapq.heappush(self._heap, (size, -self._degree[variable], self._position[variable], variable))

    def pick(self, repository: DomainRepository) -> str | None:
        """The open variable with the fewest values left, or None where every variable is settled."""
        while self._heap:
            size, _, _, variable = self._heap[0]
            held = len(repository.values[variable])
            if size != held or held <= 1:
                heapq.heappop(self._heap)
                continue
            return variable
        return None
