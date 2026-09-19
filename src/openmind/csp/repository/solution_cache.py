from openmind.csp.model.solve_statistics import SolveStatistics
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.service.memory_evictor import evict_oldest
from openmind.world.model.action import Action
from openmind.world.model.state import State

#: What a solution is kept under: the state, the limit, and the action's rules.
type SolutionKey = tuple[State, int | None, object]

#: The actions solved, with what the search did.
type Solution = tuple[tuple[Action, ...], SolveStatistics]


class SolutionCache:
    """The legal actions solved so far, per action and state, built once and given to the solver, which keeps nothing
    itself. The process's memory guard evicts the oldest when memory runs short. Its keys are this process's ids, so a
    copy sent to another process arrives empty."""

    def __init__(self) -> None:
        self._solutions: dict[SolutionKey, Solution] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __getstate__(self) -> dict[str, object]:
        return {}

    def __setstate__(self, state: dict[str, object]) -> None:
        self._solutions = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def get(self, key: SolutionKey) -> Solution | None:
        return self._solutions.get(key)

    def put(self, key: SolutionKey, solution: Solution) -> Solution:
        self._memory_guard.remembered()
        self._solutions[key] = solution
        return solution

    def memory_entries(self) -> int:
        return len(self._solutions)

    def evict_memory(self, entries: int) -> None:
        evict_oldest(self._solutions, entries)

    def clear_memory(self) -> None:
        self._solutions.clear()
