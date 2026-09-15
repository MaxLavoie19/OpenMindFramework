from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.inference.constant.inference_constant import DEFAULT_PROCESS_MEMORY, MEMORY_CHECK_INTERVAL
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.inference.service.position_view import Moves, PositionView
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.predictor.service.predictor import Predictor
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.model.state import State


class Mechanics:
    """What a domain's own rules make of a position, for generated expressions: the position as a view, its variables as
    rules read them, and the outcomes of every action a player could take in it if it were their turn. Views and moves
    are kept per process, left behind when the mechanics are copied to another process, and cleared whenever the process
    holds more memory than its share: every MEMORY_CHECK_INTERVAL entries remembered, the memory meter is read. The
    process's memory guard clears them too."""

    def __init__(
        self,
        solver: Solver,
        predictor: Predictor,
        state_namespace_mapper: StateNamespaceMapper,
        memory_meter: MemoryMeter,
    ) -> None:
        self._solver = solver
        self._predictor = predictor
        self._state_namespace_mapper = state_namespace_mapper
        self._memory_meter = memory_meter
        self._memory_share = DEFAULT_PROCESS_MEMORY
        self._remembered = 0
        self._domains: dict[int, Domain] = {}
        self._cache: dict[tuple[object, ...], object] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __getstate__(self) -> dict[str, object]:
        return {
            name: value for name, value in self.__dict__.items() if name not in ("_domains", "_cache", "_memory_guard")
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._domains = {}
        self._cache = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def memory_entries(self) -> int:
        return len(self._cache)

    def clear_memory(self) -> None:
        self.clear()

    def limit_memory(self, memory_bytes: int) -> None:
        """The share of memory a process holds before its views and moves are cleared; copies sent to other processes
        carry it."""
        self._memory_share = memory_bytes

    def clear(self) -> None:
        """Forgets every view and move this process kept."""
        self._cache.clear()

    def view(self, domain: Domain, state: State) -> PositionView:
        """The same view every time the state comes back, until the cache is cleared."""
        key = ("view", self._pin(domain), state)
        view = self._cache.get(key)
        if view is None:
            view = PositionView(self, domain, state)
            self._remember(key, view)
        return view  # type: ignore[return-value]

    def variables(self, state: State) -> dict[str, object]:
        """The state's variables as rules read them: `turn`, `cell[2, 3]`."""
        return self._state_namespace_mapper.to_namespace(state)

    def moves(self, domain: Domain, state: State, player: str) -> Moves:
        """For each action the player could take if it were their turn, its outcomes with a probability above 0, each
        outcome as a view with its probability."""
        key = ("moves", self._pin(domain), state, player)
        moves = self._cache.get(key)
        if moves is None:
            turned = self.with_turn(domain, state, player)
            moves = tuple(
                tuple(
                    (self.view(domain, outcome), probability)
                    for outcome, probability in self._predictor.predict(domain.transitions, turned, action).outcomes
                    if probability > 0
                )
                for action in self._solver.solve(domain.problem, turned)
            )
            self._remember(key, moves)
        return moves  # type: ignore[return-value]

    def with_turn(self, domain: Domain, state: State, player: str) -> State:
        """The state with the player to act replaced."""
        return State(
            tuple((name, player if name == domain.players.to_act else value) for name, value in state.variables)
        )

    def _pin(self, domain: Domain) -> int:
        """An identity for the domain that stays valid: the domain is kept alive as long as the mechanics."""
        self._domains.setdefault(id(domain), domain)
        return id(domain)

    def _remember(self, key: tuple[object, ...], value: object) -> None:
        self._remembered += 1
        if self._remembered % MEMORY_CHECK_INTERVAL == 0 and self._memory_meter.resident_bytes() > self._memory_share:
            self._cache.clear()
        self._memory_guard.remembered()
        self._cache[key] = value
