from collections.abc import Mapping

from openmind.world.mapper.variable_name_mapper import VariableNameMapper
from openmind.world.model.grid import Grid
from openmind.world.model.state import State
from openmind.world.model.value import Value

type Layout = tuple[tuple[str, object], ...]
#: A state's layout, and the bases whose indices are all whole numbers.
type Arrangement = tuple[Layout, frozenset[str]]

_PLAIN = object()


class StateNamespaceMapper:
    """Maps a state to the names a rule reads, and a namespace a script changed back to a state. A plain variable such as
    `turn` is its value; variables named with indices are gathered under their base in a dict, so cell(2,3) is
    cell[2, 3] and payoff(X) is payoff["X"]. An index written as a whole number is an int, and a base whose indices are
    all whole numbers, as many of them each, is a Grid, searchable by coordinates."""

    def __init__(self, variable_name_mapper: VariableNameMapper) -> None:
        self._variable_name_mapper = variable_name_mapper
        self._layouts: dict[tuple[str, ...], Arrangement] = {}

    def __getstate__(self) -> dict[str, object]:
        """Layouts mark plain variables with a sentinel only this process knows: a copy lays states out again."""
        return {"_variable_name_mapper": self._variable_name_mapper}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._layouts = {}

    def to_namespace(self, state: State) -> dict[str, object]:
        """A fresh namespace: changing it, or the dicts in it, doesn't change the state."""
        namespace: dict[str, object] = {}
        layout, grids = self._arrangement(state)
        for (base, key), (_, value) in zip(layout, state.variables, strict=True):
            if key is _PLAIN:
                namespace[base] = value
            else:
                indexed = namespace.get(base)
                if indexed is None:
                    indexed = namespace[base] = Grid() if base in grids else {}
                indexed[key] = value  # type: ignore[index]
        return namespace

    def to_state(self, state: State, namespace: Mapping[str, object]) -> State:
        """The state's variables with their values read back from the namespace. An index a script added under a base the
        state has is a new variable, after the state's own, in the order the script added it; an index that can't be
        written in a variable name and read back the same raises ValueError."""
        layout = self._layout(state)
        variables: list[tuple[str, Value]] = []
        known: dict[str, int] = {}
        for (base, key), (name, _) in zip(layout, state.variables, strict=True):
            if key is _PLAIN:
                variables.append((name, namespace[base]))  # type: ignore[arg-type]
            else:
                variables.append((name, namespace[base][key]))  # type: ignore[index]
                known[base] = known.get(base, 0) + 1
        for base, count in known.items():
            indexed = namespace[base]
            if len(indexed) != count:  # type: ignore[arg-type]
                keys = {key for layout_base, key in layout if layout_base == base}
                for extra, value in indexed.items():  # type: ignore[attr-defined]
                    if extra not in keys:
                        variables.append((self._added_name(base, extra), value))
        return State(tuple(variables))

    def _added_name(self, base: str, key: object) -> str:
        indices = key if isinstance(key, tuple) else (key,)
        name = self._variable_name_mapper.to_name(base, indices)
        read_back = self._variable_name_mapper.from_name(name)
        if read_back != (base, tuple(str(index) for index in indices)) or tuple(map(self._index, read_back[1])) != indices:
            raise ValueError(f"The index {key!r} of {base!r} can't be written in a variable name")
        return name

    def cells(self, state: State) -> tuple[tuple[str, object] | None, ...]:
        """For each of the state's variables, in order, its base and its index as a rule reads them, or None for a plain
        variable."""
        return tuple(None if key is _PLAIN else (base, key) for base, key in self._layout(state))

    def to_source(self, name: str) -> str:
        """How a rule reads a variable: `turn`, `cell[2, 3]`, `payoff['X']`."""
        base, texts = self._variable_name_mapper.from_name(name)
        if not texts:
            return base
        return f"{base}[{', '.join(repr(self._index(text)) for text in texts)}]"

    def _layout(self, state: State) -> Layout:
        return self._arrangement(state)[0]

    def _arrangement(self, state: State) -> Arrangement:
        names = tuple(name for name, _ in state.variables)
        arrangement = self._layouts.get(names)
        if arrangement is None:
            entries: list[tuple[str, object]] = []
            for name in names:
                base, texts = self._variable_name_mapper.from_name(name)
                if not texts:
                    entries.append((base, _PLAIN))
                else:
                    indices = tuple(self._index(text) for text in texts)
                    entries.append((base, indices[0] if len(indices) == 1 else indices))
            plain = {base for base, key in entries if key is _PLAIN}
            indexed = {base for base, key in entries if key is not _PLAIN}
            if clash := sorted(plain & indexed):
                raise ValueError(f"State variable {clash[0]!r} is both a plain and an indexed variable")
            keys: dict[str, list[object]] = {}
            for base, key in entries:
                if key is not _PLAIN:
                    keys.setdefault(base, []).append(key)
            grids = frozenset(base for base, found in keys.items() if self._whole(found))
            arrangement = self._layouts[names] = (tuple(entries), grids)
        return arrangement

    def _whole(self, keys: list[object]) -> bool:
        """Whether every key is a whole number, or every key a tuple of as many whole numbers."""
        if all(isinstance(key, int) and not isinstance(key, bool) for key in keys):
            return True
        return all(isinstance(key, tuple) for key in keys) and len({len(key) for key in keys}) == 1 and all(  # type: ignore[arg-type]
            isinstance(index, int) and not isinstance(index, bool) for key in keys for index in key  # type: ignore[attr-defined]
        )

    def _index(self, text: str) -> object:
        return int(text) if text.lstrip("-").isdecimal() else text
