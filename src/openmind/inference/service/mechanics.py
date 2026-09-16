from collections import Counter

from openmind.agent.model.domain import Domain
from openmind.csp.service.solver import Solver
from openmind.inference.constant.inference_constant import DEFAULT_PROCESS_MEMORY, MEMORY_CHECK_INTERVAL
from openmind.parallel.service.memory_evictor import evict_oldest
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.inference.service.position_view import Moves, PositionView
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.predictor.service.predictor import Predictor
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.world.model.grid import Grid
from openmind.world.model.state import State
from openmind.world.model.value import Value


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

    def evict_memory(self, entries: int) -> None:
        evict_oldest(self._cache, entries)

    def clear_memory(self) -> None:
        self.clear()

    def limit_memory(self, memory_bytes: int) -> None:
        """The share of memory a process holds before its views and moves are cleared; copies sent to other processes
        carry it."""
        self._memory_share = memory_bytes

    def clear(self) -> None:
        """Forgets every view and move this process kept."""
        self._cache.clear()

    def view(self, domain: Domain, state: State, after: PositionView | None = None) -> PositionView:
        """The same view every time the state comes back, until the cache is cleared. `after` is the position a move
        led here from, when there is one: the view lays itself out from that position's rather than from nothing."""
        key = ("view", self._pin(domain), state)
        view = self._cache.get(key)
        if view is None:
            view = PositionView(self, domain, state, after)
            self._remember(key, view)
        return view  # type: ignore[return-value]

    def variables(self, state: State) -> dict[str, object]:
        """The state's variables as rules read them: `turn`, `cell[2, 3]`."""
        return self._state_namespace_mapper.to_namespace(state)

    def variables_after(self, before: State, namespace: dict[str, object], state: State) -> dict[str, object]:
        """The state's variables as rules read them, from those of the state a move led here from: only what differs is
        written, the rest copied."""
        return self._state_namespace_mapper.to_namespace_after(before, namespace, state)

    def moves(self, domain: Domain, state: State, player: str) -> Moves:
        """For each action the player could take if it were their turn, its outcomes with a probability above 0, each
        outcome as a view with its probability."""
        key = ("moves", self._pin(domain), state, player)
        moves = self._cache.get(key)
        if moves is None:
            turned = self.with_turn(domain, state, player)
            came_from = self.view(domain, turned)
            moves = tuple(
                tuple(
                    (self.view(domain, outcome, came_from), probability)
                    for outcome, probability in self._predictor.predict(domain.transitions, turned, action).outcomes
                    if probability > 0
                )
                for action in self._solver.solve(domain.problem, turned)
            )
            self._remember(key, moves)
        return moves  # type: ignore[return-value]

    def changes(self, domain: Domain, state: State, player: str) -> dict[tuple[str, object], float]:
        """For each variable of an indexed base, by base and index as rules read them, how many of the player's actions,
        as if it were their turn, change it, each outcome weighted by its probability; variables no action changes are
        left out. Worked out once for the state and player, from the moves."""
        key = ("changes", self._pin(domain), state, player)
        changes = self._cache.get(key)
        if changes is None:
            cells = self._state_namespace_mapper.cells(state)
            found: dict[tuple[str, object], float] = {}
            for outcomes in self.moves(domain, state, player):
                for view, probability in outcomes:
                    for cell in self._changed_cells(state, view.state, cells):
                        found[cell] = found.get(cell, 0.0) + probability
            changes = found
            self._remember(key, changes)
        return changes  # type: ignore[return-value]

    def _changed_cells(
        self, before: State, after: State, cells: tuple[tuple[str, object] | None, ...]
    ) -> list[tuple[str, object]]:
        """The indexed variables whose value differs after; by the rules' names when the outcome's variables aren't the
        state's own, in the same order."""
        if len(before.variables) != len(after.variables):
            return self._changed_by_names(before, after)
        changed: list[tuple[str, object]] = []
        for (name, value), (after_name, after_value), cell in zip(before.variables, after.variables, cells, strict=True):
            if name != after_name:
                return self._changed_by_names(before, after)
            if cell is not None and value != after_value:
                changed.append(cell)
        return changed

    def _changed_by_names(self, before: State, after: State) -> list[tuple[str, object]]:
        first, second = self.variables(before), self.variables(after)
        return [
            (base, index)
            for base, cells in first.items()
            if isinstance(cells, dict)
            for index, value in cells.items()
            if not (isinstance(second.get(base), dict) and index in second[base] and second[base][index] == value)  # type: ignore[index, operator]
        ]

    def empties(self, domain: Domain) -> dict[str, Value]:
        """For each grid of the domain, the value most of its cells hold in the initial position: its empty value."""
        key = ("empties", self._pin(domain))
        empties = self._cache.get(key)
        if empties is None:
            empties = {
                base: Counter(cells.values()).most_common(1)[0][0]
                for base, cells in self.variables(domain.initial_state).items()
                if isinstance(cells, Grid) and cells
            }
            self._remember(key, empties)
        return empties  # type: ignore[return-value]

    def with_value(self, state: State, base: str, at: object, value: Value) -> State:
        """The state with the variable of `base` at index `at` set to the value; a variable it doesn't have raises
        KeyError. Worked out once for the state, variable and value."""
        key = ("with_value", state, base, at, value)
        edited = self._cache.get(key)
        if edited is None:
            cells = self._state_namespace_mapper.cells(state)
            variables = list(state.variables)
            for index, cell in enumerate(cells):
                if cell == (base, at):
                    variables[index] = (variables[index][0], value)
                    break
            else:
                raise KeyError(f"The position has no variable {base}[{at!r}]")
            edited = State(tuple(variables))
            self._remember(key, edited)
        return edited  # type: ignore[return-value]

    def cleared(self, domain: Domain, state: State, at: object) -> State:
        """The state with every grid's cell at `at` set to that grid's empty value."""
        empties = self.empties(domain)
        cells = self._state_namespace_mapper.cells(state)
        variables = [
            (name, empties[cell[0]] if cell is not None and cell[1] == at and cell[0] in empties else value)
            for (name, value), cell in zip(state.variables, cells, strict=True)
        ]
        return State(tuple(variables))

    def copied(self, state: State, source: object, target: object) -> State:
        """The state with every grid's value at `source` also placed at `target`."""
        cells = self._state_namespace_mapper.cells(state)
        held = {
            cell[0]: value for (_, value), cell in zip(state.variables, cells, strict=True) if cell is not None and cell[1] == source
        }
        variables = [
            (name, held[cell[0]] if cell is not None and cell[1] == target and cell[0] in held else value)
            for (name, value), cell in zip(state.variables, cells, strict=True)
        ]
        return State(tuple(variables))

    def alone(self, domain: Domain, state: State, at: object) -> State:
        """The state with every grid's cells set to that grid's empty value, except at `at`. Worked out once for the
        state and cell."""
        key = ("alone", self._pin(domain), state, at)
        edited = self._cache.get(key)
        if edited is None:
            empties = self.empties(domain)
            cells = self._state_namespace_mapper.cells(state)
            variables = [
                (name, empties[cell[0]] if cell is not None and cell[1] != at and cell[0] in empties else value)
                for (name, value), cell in zip(state.variables, cells, strict=True)
            ]
            edited = State(tuple(variables))
            self._remember(key, edited)
        return edited  # type: ignore[return-value]

    def with_turn(self, domain: Domain, state: State, player: str) -> State:
        """The state with the player to act replaced."""
        return State(
            tuple((name, player if name == domain.players.to_act else value) for name, value in state.variables)
        )

    def _pin(self, domain: Domain) -> int:
        """An identity for the domain that stays valid: the domain is kept alive as long as the mechanics."""
        key = id(domain)
        if key not in self._domains:
            self._domains[key] = domain
        return key

    def _remember(self, key: tuple[object, ...], value: object) -> None:
        self._remembered += 1
        if self._remembered % MEMORY_CHECK_INTERVAL == 0 and self._memory_meter.resident_bytes() > self._memory_share:
            self._cache.clear()
        self._memory_guard.remembered()
        self._cache[key] = value
