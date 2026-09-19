from collections.abc import Mapping
from types import FunctionType

from openmind.parallel.factory.memory_guard_factory import process_memory_guard
from openmind.parallel.service.memory_evictor import evict_oldest
from openmind.rule.constant.rule_constant import ALL_DIFFERENT
from openmind.rule.mapper.state_namespace_mapper import StateNamespaceMapper
from openmind.rule.model.compiled_rule import CompiledRule
from openmind.structure.model.grid import Grid
from openmind.structure.model.list import List
from openmind.structure.model.map import Map
from openmind.structure.model.scalar import Scalar
from openmind.world.model.state import State
from openmind.structure.model.value import Value


#: The data models every rule can build: `Grid.filled((3, 3), None)`, `Map.of({"X": None})`.
STRUCTURES = {"Grid": Grid, "List": List, "Map": Map, "Scalar": Scalar}


def _all_different(*values: object) -> bool:
    return len(set(values)) == len(values)


class RuleRunner:
    """Runs compiled rules against states. A value rule is called as a function of the parameters it reads, with the
    definitions' names and the state's variables as its globals, built once per state and kept. An effects rule runs as
    a module in a fresh copy of those names plus the action's parameters, and what it leaves in the state's variables is
    the next state. Every rule also sees `all_different(*values)` and the data models, `Grid`, `List`, `Map` and `Scalar`. The namespaces built per state are kept until the
    process's memory guard clears them."""

    def __init__(self, state_namespace_mapper: StateNamespaceMapper) -> None:
        self._state_namespace_mapper = state_namespace_mapper
        self._definitions: dict[CompiledRule | None, dict[str, object]] = {}
        self._namespaces: dict[
            tuple[CompiledRule | None, State, int | None], tuple[Mapping[str, object] | None, dict[str, object]]
        ] = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def __getstate__(self) -> dict[str, object]:
        """Namespaces hold functions and modules, which don't travel to other processes: a copy builds its own."""
        return {"_state_namespace_mapper": self._state_namespace_mapper}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._definitions = {}
        self._namespaces = {}
        self._memory_guard = process_memory_guard()
        self._memory_guard.register(self)

    def memory_entries(self) -> int:
        return len(self._namespaces)

    def evict_memory(self, entries: int) -> None:
        evict_oldest(self._namespaces, entries)

    def clear_memory(self) -> None:
        self._namespaces.clear()

    def value(
        self,
        compiled: CompiledRule,
        state: State,
        parameters: Mapping[str, object] | None = None,
        names: Mapping[str, object] | None = None,
    ) -> object:
        """The rule's value; a parameter it reads that isn't given raises KeyError. `names` adds names the rule can read,
        such as functions bound to this state; the same mapping object should be passed for the same state, since the
        namespace built from it is kept. A value rule must not change the names it reads."""
        function = FunctionType(compiled.code, self._namespace(compiled.definitions, state, names))
        given = parameters or {}
        return function(*(given[name] for name in compiled.arguments))

    def apply(self, compiled: CompiledRule, state: State, parameters: Mapping[str, Value] | None = None) -> State:
        """The state after running the script; a parameter named like a state variable raises ValueError."""
        namespace = self._state_namespace_mapper.to_namespace(state)
        if parameters:
            if clash := sorted(set(parameters) & set(namespace)):
                raise ValueError(f"Parameter {clash[0]!r} has the name of a state variable")
            namespace = {**self._definitions_namespace(compiled.definitions), **namespace, **parameters}
        else:
            namespace = {**self._definitions_namespace(compiled.definitions), **namespace}
        exec(compiled.code, namespace)
        return self._state_namespace_mapper.to_state(state, namespace)

    def _namespace(
        self, definitions: CompiledRule | None, state: State, names: Mapping[str, object] | None
    ) -> dict[str, object]:
        key = (definitions, state, None if names is None else id(names))
        entry = self._namespaces.get(key)
        if entry is None or entry[0] is not names:
            variables = self._state_namespace_mapper.to_namespace(state)
            if names is not None and (clash := sorted(set(names) & set(variables))):
                raise ValueError(f"Name {clash[0]!r} is both a state variable and an added name")
            namespace = {**self._definitions_namespace(definitions), **variables, **(names or {})}
            self._memory_guard.remembered()
            self._namespaces[key] = (names, namespace)
            return namespace
        return entry[1]

    def _definitions_namespace(self, definitions: CompiledRule | None) -> dict[str, object]:
        namespace = self._definitions.get(definitions)
        if namespace is None:
            namespace = {ALL_DIFFERENT: _all_different, **STRUCTURES}
            if definitions is not None:
                exec(definitions.code, namespace)
            self._definitions[definitions] = namespace
        return namespace
