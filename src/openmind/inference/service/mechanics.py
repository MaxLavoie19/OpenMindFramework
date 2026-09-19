from openmind.inference.constant.inference_constant import DEFAULT_PROCESS_MEMORY, MEMORY_CHECK_INTERVAL
from openmind.inference.service.position_view import Moves, PositionView
from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.service.memory_evictor import evict_oldest
from openmind.parallel.service.memory_meter import MemoryMeter
from openmind.rbs.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rbs.service.rule_based_system import RuleBasedSystem
from openmind.structure.model.data_model import DataModel
from openmind.structure.model.grid import Grid
from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar
from openmind.world.model.state import State
from openmind.structure.model.value import Value


class Mechanics:
    """What a domain's own rules make of a position, for generated expressions: the position as a view, its models as
    rules read them, and the outcomes of every action a player could take in it if it were their turn. Views and moves
    are kept per process, left behind when the mechanics are copied to another process, and cleared whenever the process
    holds more memory than its share: every MEMORY_CHECK_INTERVAL entries remembered, the memory meter is read. The
    process's memory guard clears them too."""

    def __init__(
        self,
        state_namespace_mapper: StateNamespaceMapper,
        memory_meter: MemoryMeter,
    ) -> None:
        self._state_namespace_mapper = state_namespace_mapper
        self._memory_meter = memory_meter
        self._memory_share = DEFAULT_PROCESS_MEMORY
        self._remembered = 0
        self._games: dict[int, Domain] = {}
        self._cache: dict[tuple[object, ...], object] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __getstate__(self) -> dict[str, object]:
        return {
            name: value for name, value in self.__dict__.items() if name not in ("_games", "_cache", "_memory_guard")
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._games = {}
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

    def view(self, rbs: RuleBasedSystem, state: State, after: PositionView | None = None) -> PositionView:
        """The same view every time the state comes back, until the cache is cleared. `after` is the position a move
        led here from, when there is one: the view lays itself out from that position's rather than from nothing."""
        key = ("view", self._pin(rbs), state)
        view = self._cache.get(key)
        if view is None:
            view = PositionView(self, rbs, state, after)
            self._remember(key, view)
        return view  # type: ignore[return-value]

    def variables(self, state: State) -> dict[str, object]:
        """The state's models as rules read them: a scalar as its value, `turn`; a grid, list or map as itself,
        `cell[2, 3]`."""
        return self._state_namespace_mapper.to_namespace(state)

    def variables_after(self, before: State, namespace: dict[str, object], state: State) -> dict[str, object]:
        """The state's models as rules read them, for a state a move led to from `before`, whose namespace is given."""
        return self._state_namespace_mapper.to_namespace_after(before, namespace, state)

    def moves(self, rbs: RuleBasedSystem, state: State, player: str) -> Moves:
        """For each action the player could take if it were their turn, its outcomes with a probability above 0, each
        outcome as a view with its probability."""
        key = ("moves", self._pin(rbs), state, player)
        moves = self._cache.get(key)
        if moves is None:
            turned = self.with_turn(rbs, state, player)
            came_from = self.view(rbs, turned)
            moves = tuple(
                tuple(
                    (self.view(rbs, outcome, came_from), probability)
                    for outcome, probability in rbs.outcomes(turned, action).outcomes
                    if probability > 0
                )
                for action in rbs.actions(turned)
            )
            self._remember(key, moves)
        return moves  # type: ignore[return-value]

    def changes(self, rbs: RuleBasedSystem, state: State, player: str) -> dict[tuple[str, object], float]:
        """For each part of the state, how many of the player's actions, as if it were their turn, change it, each
        outcome weighted by its probability; parts no action changes are left out. A grid's cell is keyed by the grid's
        name and its coordinates, a map's entry by the map's name and its key, and a scalar or a list, changed as a
        whole, by its name and None. Worked out once for the state and player, from the moves."""
        key = ("changes", self._pin(rbs), state, player)
        changes = self._cache.get(key)
        if changes is None:
            found: dict[tuple[str, object], float] = {}
            for outcomes in self.moves(rbs, state, player):
                for view, probability in outcomes:
                    for part in self._changed_parts(state, view.state):
                        found[part] = found.get(part, 0.0) + probability
            changes = found
            self._remember(key, changes)
        return changes  # type: ignore[return-value]

    def _changed_parts(self, before: State, after: State) -> list[tuple[str, object]]:
        """The parts of the state whose value differs after: grid cells by coordinates, map entries by key, and scalars
        and lists as a whole. A model the state before doesn't have isn't a change of the state's own parts; a cell or an
        entry only one of the two has counts as changed."""
        changed: list[tuple[str, object]] = []
        for name, model in before.models:
            if not after.has(name):
                changed.extend(self._parts(name, model))
                continue
            held = after.model(name)
            if held == model:
                continue
            if isinstance(model, Grid) and isinstance(held, Grid) and model.shape == held.shape:
                changed.extend(
                    (name, where)
                    for where, value, value_after in zip(model.coordinates(), model.cells, held.cells, strict=True)
                    if value != value_after
                )
            elif isinstance(model, Map) and isinstance(held, Map):
                keys = dict.fromkeys((*model.keys(), *held.keys()))
                changed.extend(
                    (name, key) for key in keys if key not in model or key not in held or model[key] != held[key]
                )
            else:
                changed.extend(self._parts(name, model))
        return changed

    def _parts(self, name: str, model: DataModel) -> list[tuple[str, object]]:
        if isinstance(model, Grid):
            return [(name, where) for where in model.coordinates()]
        if isinstance(model, Map):
            return [(name, key) for key in model.keys()]
        return [(name, None)]

    def empties(self, rbs: RuleBasedSystem) -> dict[str, Value]:
        """Each grid's empty value, as the domain declares it."""
        return dict(rbs.empties())

    def with_value(self, state: State, base: str, at: object, value: Value) -> State:
        """The state with the cell of the grid `base` at `at`, or the entry of the map `base` at the key `at`, set to
        the value; a cell or an entry the state doesn't have raises KeyError. Worked out once for the state, cell and
        value."""
        key = ("with_value", state, base, at, value)
        edited = self._cache.get(key)
        if edited is None:
            model = state.model(base) if state.has(base) else None
            if isinstance(model, Grid) and isinstance(at, tuple) and model.inside(at):
                edited = state.with_model(base, model.placed(at, value))
            elif isinstance(model, Map) and at in model:
                edited = state.with_model(base, model.with_item(at, value))  # type: ignore[arg-type]
            else:
                raise KeyError(f"The position has no cell {base}[{at!r}]")
            self._remember(key, edited)
        return edited  # type: ignore[return-value]

    def cleared(self, rbs: RuleBasedSystem, state: State, at: object) -> State:
        """The state with every grid's cell at `at` set to that grid's empty value. A grid without a declared empty value
        raises KeyError."""
        empties = self._declared(rbs, state)
        edited = state
        for name, grid in self._grids(state):
            if isinstance(at, tuple) and grid.inside(at):
                edited = edited.with_model(name, grid.placed(at, empties[name]))
        return edited

    def copied(self, state: State, source: object, target: object) -> State:
        """The state with every grid's value at `source` also placed at `target`, in the grids having both cells."""
        edited = state
        for name, grid in self._grids(state):
            if isinstance(source, tuple) and isinstance(target, tuple) and grid.inside(source) and grid.inside(target):
                edited = edited.with_model(name, grid.placed(target, grid.at(source)))
        return edited

    def alone(self, rbs: RuleBasedSystem, state: State, at: object) -> State:
        """The state with every grid's cells set to that grid's empty value, except at `at`. Worked out once for the
        state and cell. A grid without a declared empty value raises KeyError."""
        key = ("alone", self._pin(rbs), state, at)
        edited = self._cache.get(key)
        if edited is None:
            empties = self._declared(rbs, state)
            edited = state
            for name, grid in self._grids(state):
                emptied = Grid.filled(grid.shape, empties[name], grid.aliases, grid.directions)
                if isinstance(at, tuple) and grid.inside(at):
                    emptied = emptied.placed(at, grid.at(at))
                edited = edited.with_model(name, emptied)
            self._remember(key, edited)
        return edited  # type: ignore[return-value]

    def _grids(self, state: State) -> list[tuple[str, Grid]]:
        return [(name, model) for name, model in state.models if isinstance(model, Grid)]

    def _declared(self, rbs: RuleBasedSystem, state: State) -> dict[str, Value]:
        """The empty values of the state's grids; a grid the domain declares none for raises KeyError."""
        empties = self.empties(rbs)
        grids = [name for name, _ in self._grids(state)]
        missing = [name for name in grids if name not in empties]
        if missing:
            raise KeyError(f"{rbs.context} declares no empty value for {', '.join(missing)}")
        return {name: empties[name] for name in grids}

    def with_turn(self, rbs: RuleBasedSystem, state: State, player: str) -> State:
        """The state with the player to act replaced, where a scalar names that player; flags of players acting at once
        are left as they are."""
        to_act = rbs.players().to_act
        if state.has(to_act) and isinstance(state.model(to_act), Scalar):
            return state.with_model(to_act, player)
        return state

    def _pin(self, rbs: RuleBasedSystem) -> int:
        """An identity for the domain that stays valid: the domain is kept alive as long as the mechanics."""
        key = id(rbs)
        if key not in self._games:
            self._games[key] = rbs
        return key

    def _remember(self, key: tuple[object, ...], value: object) -> None:
        self._remembered += 1
        if self._remembered % MEMORY_CHECK_INTERVAL == 0 and self._memory_meter.resident_bytes() > self._memory_share:
            self._cache.clear()
        self._memory_guard.remembered()
        self._cache[key] = value
